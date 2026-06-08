"""Quick end-to-end test: process 50 samples, validate distributions, train, print topics.

Usage:
    python scripts/test_pipeline.py
"""

import os
import sys
import json
import shutil
import tempfile

import numpy as np
import torch
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loaders import get_local_dataset, PROCESSED_DATA_DIR
from data.tokenization import tokenize_dataset_batch
from data.processing_utils import (
    collate_fn, extract_embeddings,
    write_batch_to_parquet, save_hf_dataset_from_parquet,
)
from data.dataset.ctm_dataset import get_ctm_dataset_from_processed_data
from models.ctm import GenerativeTM

MAX_SAMPLES = 50
NUM_TOPICS = 5
NUM_EPOCHS = 30
VOCAB_SIZE = 2000
DATASET_PATH = "data/raw_data/tweet_topic.tsv"

MODELS = [
    ("microsoft/Phi-3-mini-128k-instruct", 4),
    ("Qwen/Qwen3.5-0.8B", 8),
]


def process_subset(model_name, batch_size, dataset, vocab, tokenizer, save_name):
    """Process MAX_SAMPLES through the LLM and save as HF dataset."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_kwargs = dict(dtype=torch.bfloat16)
    try:
        import flash_attn  # noqa: F401
        model_kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        model_kwargs["attn_implementation"] = "sdpa"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, **model_kwargs
        ).eval()
    except (ImportError, Exception):
        model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs
        ).eval()
    model.to(device)

    # Build vocab prefix tokens
    vocab_token_ids = [tokenizer.encode(f" {word}", add_special_tokens=False) for word in vocab]
    raw_prefixes = [ids[0] for ids in vocab_token_ids]
    if len(set(raw_prefixes)) == 1 and all(len(ids) > 1 for ids in vocab_token_ids):
        print(f"  Detected shared space-prefix token {raw_prefixes[0]}; using index 1.")
        vocab_token_prefix = [ids[1] for ids in vocab_token_ids]
    else:
        vocab_token_prefix = raw_prefixes

    n_unique = len(set(vocab_token_prefix))
    print(f"  Unique prefix token IDs: {n_unique}/{len(vocab)}")

    vocab_set = set(vocab)
    from llm import jinja_template_manager  # noqa: E402

    from torch.utils.data import DataLoader

    subset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(
            batch, tokenizer, "text", "label",
            vocab_set, "instructions/default.jinja", "document_topic_distribution.jinja"
        ),
        num_workers=0,
    )

    temp_dir = tempfile.mkdtemp()
    parquet_files = []
    total = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        bs = input_ids.shape[0]

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, use_cache=False, output_hidden_states=True)

        logits_all = outputs.logits
        next_token_logits = logits_all[:, -1, :]
        next_words = [tokenizer.decode(tid) for tid in torch.argmax(next_token_logits, dim=-1)]

        for b in range(bs):
            emb = extract_embeddings(outputs.hidden_states, attention_mask, b, None, "last")
            vl = next_token_logits[b, vocab_token_prefix]
            nl = np.array([vl[i].float().cpu().item() for i in range(len(vocab))], dtype=np.float32).tolist()

            bow = batch["bow_lines"][b]
            label = batch["labels"][b]

            ex = {
                "id": batch["ids"][b],
                "context": batch["contexts"][b],
                "next_word": next_words[b],
                "next_word_logits": nl,
                "input_embeddings": emb,
                "bow": bow,
            }
            if label is not None:
                ex["label"] = label

            pf = write_batch_to_parquet([ex], total, temp_dir)
            if pf:
                parquet_files.append(pf)
            total += 1

        del outputs, logits_all, next_token_logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    local_path = os.path.join(PROCESSED_DATA_DIR, save_name)
    if os.path.exists(local_path):
        shutil.rmtree(local_path)

    metadata = {"args": {"model_name": model_name, "dataset": DATASET_PATH}, "vocab_size": len(vocab)}
    save_hf_dataset_from_parquet(parquet_files, local_path, vocab, metadata, save_name)
    shutil.rmtree(temp_dir)

    print(f"  Saved {total} samples to {local_path}")
    return local_path


def validate(data_path, vocab):
    """Check that logit distributions are meaningful, not uniform."""
    ds = load_from_disk(data_path)
    vocab_size = len(vocab)
    max_entropy = np.log(vocab_size)
    passed = True

    entropies = []
    for i, sample in enumerate(ds):
        logits = np.array(sample["next_word_logits"], dtype=np.float64)
        if logits.shape[0] != vocab_size:
            print(f"  FAIL sample {i}: logit len {logits.shape[0]} != {vocab_size}")
            passed = False
            continue
        if not np.all(np.isfinite(logits)):
            print(f"  FAIL sample {i}: NaN/Inf in logits")
            passed = False
            continue
        probs = softmax(logits)
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        entropies.append(entropy)

        if np.allclose(np.array(sample["input_embeddings"], dtype=np.float64).flatten()[-100:], 0.0):
            print(f"  FAIL sample {i}: zero embedding")
            passed = False

    mean_ent = np.mean(entropies)
    min_ent = np.min(entropies)
    max_ent_obs = np.max(entropies)
    ratio = mean_ent / max_entropy

    print(f"  Entropy: mean={mean_ent:.4f} min={min_ent:.4f} max={max_ent_obs:.4f}  (uniform={max_entropy:.4f})")
    print(f"  Entropy/MaxEntropy ratio: {ratio:.4f}")

    if np.isclose(mean_ent, max_entropy, atol=0.01):
        print(f"  FAIL: Distribution is UNIFORM -- logits are all the same value!")
        passed = False
    elif ratio > 0.95:
        print(f"  WARNING: Near-uniform distribution")
    else:
        print(f"  OK: Distributions show meaningful variation")

    # Print top predicted words for first 3 samples
    for i in range(min(3, len(ds))):
        logits = np.array(ds[i]["next_word_logits"], dtype=np.float64)
        probs = softmax(logits)
        top_idx = np.argsort(probs)[::-1][:10]
        top_words = [f"{vocab[j]}({probs[j]:.3f})" for j in top_idx]
        print(f"  Sample {i} top-10: {', '.join(top_words)}")

    return passed


def train_and_show_topics(data_path, vocab):
    """Train a small GenerativeTM and print topic words."""
    ds = load_from_disk(data_path)
    ctm_dataset = get_ctm_dataset_from_processed_data(ds, vocab)

    tm = GenerativeTM(
        vocab_size=len(vocab),
        embedding_size=ctm_dataset.x_embeddings.shape[1],
        num_topics=NUM_TOPICS,
        num_epochs=NUM_EPOCHS,
        batch_size=16,
        top_words=10,
        temperature=3.0,
    )
    tm.fit(ctm_dataset, verbose=False)
    info = tm.get_info()

    print(f"\n  --- Discovered Topics (K={NUM_TOPICS}) ---")
    for i, topic in enumerate(info["topics"]):
        print(f"  Topic {i}: {', '.join(topic)}")

    all_words = [w for t in info["topics"] for w in t]
    unique = len(set(all_words))
    total = len(all_words)
    print(f"\n  Topic diversity: {unique}/{total} unique words ({unique/total:.1%})")

    return True


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    all_passed = True

    for model_name, batch_size in MODELS:
        model_basename = os.path.basename(model_name)
        save_name = f"_test_{model_basename}_vocab_{VOCAB_SIZE}_last"

        print(f"\n{'#'*70}")
        print(f"# MODEL: {model_name}  ({MAX_SAMPLES} samples)")
        print(f"{'#'*70}")

        # Load raw data and build vocab (reuse cache if available)
        cache_dir = os.path.join("data", ".cache", save_name)
        vocab_cache = os.path.join(cache_dir, "vocab.json")
        tok_cache = os.path.join(cache_dir, "tokenized_dataset")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        if os.path.exists(tok_cache) and os.path.exists(vocab_cache):
            print("  Loading cached tokenized dataset...")
            dataset = load_from_disk(tok_cache)
            with open(vocab_cache) as f:
                vocab = json.load(f)
        else:
            print("  Tokenizing dataset...")
            dataset = get_local_dataset(DATASET_PATH)
            dataset = dataset.map(
                lambda x: tokenize_dataset_batch(x, tokenizer, "text"),
                batched=True, batch_size=1000, num_proc=1,
            )
            os.makedirs(cache_dir, exist_ok=True)
            dataset.save_to_disk(tok_cache)
            all_tokens = [w for wl in dataset["words"] for w in wl]
            vocab = list(set(w for w, _ in Counter(all_tokens).most_common(VOCAB_SIZE)))
            with open(vocab_cache, "w") as f:
                json.dump(vocab, f)

        print(f"  Vocab: {len(vocab)} words, Dataset: {len(dataset)} total, using {MAX_SAMPLES}")

        # Step 1: Process subset
        print("\n  [1/3] Processing subset through LLM...")
        data_path = process_subset(model_name, batch_size, dataset, vocab, tokenizer, save_name)

        # Step 2: Validate distributions
        print("\n  [2/3] Validating logit distributions...")
        valid = validate(data_path, vocab)
        if not valid:
            print(f"\n  FAILED validation for {model_name}")
            all_passed = False
            continue

        # Step 3: Train and show topics
        print("\n  [3/3] Training topic model...")
        train_and_show_topics(data_path, vocab)

        # Cleanup test data
        test_path = os.path.join(PROCESSED_DATA_DIR, save_name)
        if os.path.exists(test_path):
            shutil.rmtree(test_path)

    print(f"\n{'='*70}")
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print(f"{'='*70}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
