"""Main script for processing datasets for topic modeling."""

import os
import json
import argparse
import time
import tempfile
import shutil
from collections import Counter
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from data.loaders import get_hf_dataset, get_local_dataset, PROCESSED_DATA_DIR
from data.tokenization import tokenize_dataset_batch
from data.processing_utils import (
    get_device_info,
    collate_fn,
    extract_embeddings,
    write_batch_to_parquet,
    save_hf_dataset_from_parquet,
    upload_dataset_to_hub,
)
from settings import settings

def main(args):
    """Main processing function."""
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Use left padding for batched processing to ensure the last token is aligned
    tokenizer.padding_side = 'left'
    
    tokenized_dataset_path = os.path.join(args.cache_path, 'tokenized_dataset')
    vocab_path = os.path.join(args.cache_path, 'vocab.json')

    # Load pre-existing vocab if provided
    if args.vocab_path:
        with open(args.vocab_path, 'r') as f:
            vocab = json.load(f)
    else:
        vocab = None

    # Load or create tokenized dataset
    if os.path.exists(tokenized_dataset_path) and os.path.exists(vocab_path):
        print(f"Loading preprocessed dataset from {tokenized_dataset_path}")
        dataset = load_from_disk(tokenized_dataset_path)
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
    else:
        # Load raw dataset
        if os.path.exists(args.dataset):
            dataset = get_local_dataset(args.dataset)
        else:
            dataset = get_hf_dataset(args.dataset, args.split)

        # Tokenize dataset
        dataset = dataset.map(
            lambda x: tokenize_dataset_batch(
                x,
                tokenizer,
                args.content_key,
                single_token_only=args.single_token_only,
                vocab=vocab
            ),
            batched=True,
            batch_size=1000,
            num_proc=1
        )
        dataset.save_to_disk(tokenized_dataset_path)
        
        # Create vocab based on word frequency if not provided
        if vocab is None:
            all_tokens = [word for tokens_list in dataset['words'] for word in tokens_list]
            counter = Counter(all_tokens)
            # Visualize top 25 most frequent words
            for word, freq in counter.most_common(25):
                print(word, freq)
            vocab = list(set(word for word, _ in counter.most_common(args.vocab_size)))

        with open(vocab_path, 'w') as f:
            json.dump(vocab, f)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).eval()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Prepare vocab token info
    vocab_token_ids = [tokenizer.encode(f" {word}", add_special_tokens=False) for word in vocab]
    vocab_token_prefix = [ids[0] for ids in vocab_token_ids]
    token_lengths = [len(token_ids) for token_ids in vocab_token_ids]
    single_token_word_idx = [i for i, token_len in enumerate(token_lengths) if token_len == 1]
    multi_token_word_idx = [i for i, token_len in enumerate(token_lengths) if token_len > 1]

    # Validation
    if (len(single_token_word_idx) + len(multi_token_word_idx)) != len(vocab):
        raise ValueError(
            "The total number of single-token and multi-token words does not match the vocabulary size.")

    if args.word_prob_method == 'prefix' and len(vocab_token_prefix) > len(set(vocab_token_prefix)):
        print(
            f"Warning: Vocab token prefix is not unique, {len(vocab_token_prefix) - len(set(vocab_token_prefix))} duplicates.",
            "Consider using 'product' method to compute word probabilities."
        )
    
    if args.single_token_only and len(multi_token_word_idx) > 0:
        raise ValueError("Single token only is set to True, but there are multi-token words in the vocabulary.")
    
    # Create vocab set for efficient lookup
    vocab_set = set(vocab)
    
    # Create DataLoader for batched processing
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(
            batch, tokenizer, args.content_key, args.label_key,
            vocab_set, args.instruction_template, args.prompt_template
        ),
        num_workers=0
    )
    
    # Get device information
    device_info = get_device_info(device)
    print(f"\nDevice info: {device_info['device_type']}")
    if device_info.get('cuda_available') and device.type == 'cuda':
        print(f"  GPU: {device_info.get('gpu_name', 'Unknown')}")
    
    # Create temp directory for incremental Arrow file writes (to avoid OOM)
    temp_dir = tempfile.mkdtemp(prefix='topic_model_processing_')
    parquet_files = []
    total_examples = 0
    has_labels = None  # Will be determined from first batch
    
    # Configure batch accumulation for efficient writes
    write_batch_size = 500  # Write to disk every N examples
    accumulated_examples = []
    
    # Start timer
    start_time = time.time()
    start_datetime = datetime.now().isoformat()
    
    try:
        batch_write_num = 0
        
        for batch in tqdm(dataloader, desc="Processing batches"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            contexts = batch['contexts']
            ids = batch['ids']
            labels = batch['labels']
            bow_lines = batch['bow_lines']
            batch_size = input_ids.shape[0]
            
            # Compute the probabilities for all examples in the batch
            with torch.no_grad():
                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=True,
                )
            
            logits = outputs.logits
            next_token_logits = logits[:, -1, :]
            next_words = tokenizer.batch_decode(torch.argmax(next_token_logits, dim=-1))
            
            # Process each example in the batch
            for b_idx in range(batch_size):
                context = contexts[b_idx]
                example_id = ids[b_idx]
                example_label = labels[b_idx]
                example_bow = bow_lines[b_idx]
                example_next_token_logits = next_token_logits[b_idx:b_idx+1, :]
                
                # Extract embeddings using helper function
                embeddings = extract_embeddings(
                    outputs.hidden_states,
                    attention_mask,
                    b_idx,
                    args.hidden_state_layer,
                    args.embedding_method
                )
                
                # Compute next word logits for words in the vocab
                vocab_logits = example_next_token_logits[0, vocab_token_prefix]
                all_logits = {vocab[i]: vocab_logits[i].item() for i in range(len(vocab))}
                next_word_logits = [all_logits[word] for word in vocab]

                # Convert logits to float32 to save storage space
                next_word_logits = np.array(next_word_logits, dtype=np.float32).tolist()

                processed_example = {
                    'id': example_id,
                    'context': context,
                    'next_word': next_words[b_idx],
                    'next_word_logits': next_word_logits,
                    'input_embeddings': embeddings,
                    'bow': example_bow,
                }
                if example_label is not None:
                    processed_example['label'] = example_label
                    if has_labels is None:
                        has_labels = True
                        first_label_type = type(example_label)
                elif has_labels is None:
                    has_labels = False
                
                accumulated_examples.append(processed_example)
                total_examples += 1
                
                # Write to disk when batch is full
                if len(accumulated_examples) >= write_batch_size:
                    parquet_path = write_batch_to_parquet(accumulated_examples, batch_write_num, temp_dir)
                    if parquet_path:
                        parquet_files.append(parquet_path)
                    accumulated_examples = []
                    batch_write_num += 1
            
            # Clear CUDA cache periodically to free memory
            if torch.cuda.is_available():
                del outputs, logits, next_token_logits
                torch.cuda.empty_cache()
        
        # Write remaining examples
        if accumulated_examples:
            parquet_path = write_batch_to_parquet(accumulated_examples, batch_write_num, temp_dir)
            if parquet_path:
                parquet_files.append(parquet_path)
            accumulated_examples = []

        # End timer
        end_time = time.time()
        end_datetime = datetime.now().isoformat()
        processing_time_seconds = end_time - start_time
        
        print(f"\nProcessing completed in {processing_time_seconds:.2f} seconds ({processing_time_seconds/60:.2f} minutes)")
        
        # Free GPU memory before saving
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Create metadata
        metadata = {
            "args": {k: v for k, v in vars(args).items() if not k.startswith('_')},
            "timing": {
                "start_datetime": start_datetime,
                "end_datetime": end_datetime,
                "processing_time_seconds": processing_time_seconds,
                "num_examples": total_examples,
                "examples_per_second": total_examples / processing_time_seconds if processing_time_seconds > 0 else 0,
            },
            "device_info": device_info,
            "vocab_size": len(vocab),
        }
        
        # Save directly to HuggingFace dataset format using memory-efficient streaming
        print(f"\nSaving dataset with {total_examples} examples ({len(parquet_files)} Parquet files)...")
        
        local_path = os.path.join(PROCESSED_DATA_DIR, args.save_name)
        save_hf_dataset_from_parquet(
            parquet_files=parquet_files,
            output_dir=local_path,
            vocab=vocab,
            metadata=metadata,
            dataset_name=args.save_name,
            description=f"Processed topic modeling dataset from {args.dataset}",
        )
        
        print(f"Dataset saved to: {local_path}")
        
        # Upload to HuggingFace Hub if repo name is provided
        if args.hf_repo_name:
            upload_dataset_to_hub(
                local_path=local_path,
                hf_repo_name=args.hf_repo_name,
                private=args.hf_private,
            )
    
    finally:
        # Clean up temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process datasets for topic modeling")
    parser.add_argument('--dataset', type=str, default='fancyzhx/dbpedia_14')
    parser.add_argument('--vocab_path', type=str, default=None)
    parser.add_argument('--content_key', type=str, default='content')
    parser.add_argument('--label_key', type=str, default=None)
    parser.add_argument('--id_key', type=str, default='id')
    parser.add_argument('--split', type=str, default='all')
    parser.add_argument('--vocab_size', type=int, default=2000)
    parser.add_argument('--model_name', type=str, default='baidu/ERNIE-4.5-0.3B-PT')
    parser.add_argument('--single_token_only', action='store_true', help="Only include single token words")
    parser.add_argument('--prompt_template', type=str, default="document_topic_distribution.jinja")
    parser.add_argument('--instruction_template', type=str, default="instructions/default.jinja")
    parser.add_argument('--word_prob_method', type=str, default='prefix', choices=['prefix', 'product'])
    parser.add_argument('--hidden_state_layer', type=int, default=None, help="Hidden state layer to save (default: all)")
    parser.add_argument('--embedding_method', type=str, default='last', choices=['mean', 'last'])
    parser.add_argument('--save_name', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hf_private', action='store_true', help="Make HuggingFace dataset private")
    parser.add_argument('--no_upload', action='store_true', help="Skip uploading to HuggingFace Hub")
    args = parser.parse_args()
    
    print(f'Processing dataset "{args.dataset}"')

    if args.save_name is None:
        args.save_name = f"{os.path.basename(args.dataset).split('.')[0]}_{os.path.basename(args.model_name)}_vocab_{args.vocab_size}_{args.embedding_method}"

    # Set HuggingFace repo name (None if --no_upload is set)
    if args.no_upload:
        args.hf_repo_name = None
    else:
        args.hf_repo_name = f"{settings.hf_username}/{args.save_name}"

    print(f'Processing dataset "{args.dataset}" with save name "{args.save_name}"')
    
    # Use a cache directory for intermediate tokenized data
    args.cache_path = os.path.join('./data', '.cache', args.save_name)
    if not os.path.exists(args.cache_path):
        os.makedirs(args.cache_path, exist_ok=True)

    main(args)
