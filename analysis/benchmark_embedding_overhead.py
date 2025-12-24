"""
Benchmark script to compare computational overhead of:
1. Embedding models used in baseline topic models (Sentence-BERT variants)
2. LLM-based processing used in our approach (ERNIE, Llama models)

This addresses concerns about the added computational overhead of LLM-based
training target construction vs. the embedding step required by baselines
like ZeroShotTM, CombinedTM, BERTopic, and FASTopic.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict, field
import warnings

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress some warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    model_name: str
    model_type: str  # 'embedding' or 'llm'
    model_size_mb: float
    num_parameters: int
    num_documents: int
    total_time_seconds: float
    docs_per_second: float
    time_per_doc_ms: float
    batch_size: int
    device: str
    dataset_name: str = "synthetic"
    avg_seq_length: float = 0.0
    flops_per_doc: float = 0.0  # FLOPs per document (forward pass)
    total_flops: float = 0.0  # Total FLOPs for all documents
    tflops_per_second: float = 0.0  # TFLOPs/s throughput
    gpu_name: Optional[str] = None
    gpu_memory_mb: Optional[float] = None
    notes: str = ""


@dataclass
class DatasetStats:
    """Container for dataset statistics."""
    name: str
    num_documents: int
    avg_doc_length_chars: float
    avg_doc_length_words: float
    min_doc_length_chars: int
    max_doc_length_chars: int
    min_doc_length_words: int
    max_doc_length_words: int
    num_labels: Optional[int] = None
    label_distribution: Optional[dict] = None


def get_device_info(device: torch.device) -> dict:
    """Get device information."""
    info = {"device": str(device)}
    if device.type == "cuda" and torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(device)
        info["gpu_memory_total_mb"] = torch.cuda.get_device_properties(device).total_memory / 1024**2
    return info


def count_parameters(model) -> int:
    """Count total parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def estimate_model_size_mb(model) -> float:
    """Estimate model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024**2


def estimate_transformer_flops(num_params: int, seq_length: int, is_encoder: bool = True) -> float:
    """
    Estimate FLOPs for a transformer forward pass.
    
    For transformers, FLOPs ≈ 2 * P * S for the forward pass, where:
    - P = number of parameters
    - S = sequence length
    
    This is a simplified estimate. More precise calculations would consider:
    - Self-attention: O(n^2 * d) for sequence length n and hidden dim d
    - FFN: O(n * d^2) 
    - But 2*P*S is a reasonable approximation for comparing models
    
    Args:
        num_params: Number of model parameters
        seq_length: Input sequence length
        is_encoder: True for encoder models (BERT-like), False for decoder (GPT-like)
        
    Returns:
        Estimated FLOPs for one forward pass
    """
    # Base FLOPs: 2 * params * seq_length (matrix multiplications dominate)
    # For decoder models with causal attention, slightly less but we use same estimate
    flops = 2 * num_params * seq_length
    return flops


def generate_synthetic_documents(num_docs: int, min_length: int = 50, max_length: int = 200) -> list[str]:
    """Generate synthetic documents for benchmarking."""
    np.random.seed(42)
    
    # Sample words (mimicking real document content)
    sample_words = [
        "the", "a", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "machine", "learning", "model", "data", "analysis", "research", "study",
        "method", "approach", "results", "paper", "algorithm", "system", "process",
        "information", "knowledge", "understanding", "problem", "solution", "technique",
        "neural", "network", "deep", "training", "optimization", "performance",
        "accuracy", "precision", "recall", "evaluation", "metrics", "benchmark",
        "computer", "science", "technology", "software", "hardware", "computing",
        "natural", "language", "processing", "text", "document", "corpus",
        "topic", "semantic", "representation", "embedding", "vector", "feature",
    ]
    
    documents = []
    for _ in range(num_docs):
        length = np.random.randint(min_length, max_length)
        doc = " ".join(np.random.choice(sample_words, size=length, replace=True))
        documents.append(doc)
    
    return documents


def load_real_datasets(content_key: str = "text") -> dict[str, tuple[list[str], Optional[list]]]:
    """Load real datasets for benchmarking.
    
    Returns:
        Dictionary mapping dataset name to (documents, labels) tuple
    """
    from data.loaders import get_hf_dataset, get_local_dataset
    
    datasets = {}
    
    # 1. SetFit/20_newsgroups
    print("\nLoading SetFit/20_newsgroups...")
    try:
        ds = get_hf_dataset("SetFit/20_newsgroups", split="train")
        documents = list(ds[content_key])
        labels = list(ds["label"]) if "label" in ds.column_names else None
        datasets["20_newsgroups"] = (documents, labels)
        print(f"  Loaded {len(documents)} documents")
    except Exception as e:
        print(f"  Failed to load 20_newsgroups: {e}")
    
    # 2. stackoverflow.tsv (local)
    print("\nLoading stackoverflow.tsv...")
    try:
        stackoverflow_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "raw_data", "stackoverflow.tsv"
        )
        ds = get_local_dataset(stackoverflow_path)
        documents = list(ds[content_key])
        labels = list(ds["label"]) if "label" in ds.column_names else None
        datasets["stackoverflow"] = (documents, labels)
        print(f"  Loaded {len(documents)} documents")
    except Exception as e:
        print(f"  Failed to load stackoverflow: {e}")
    
    # 3. tweet_topic.tsv (local)
    print("\nLoading tweet_topic.tsv...")
    try:
        tweet_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "raw_data", "tweet_topic.tsv"
        )
        ds = get_local_dataset(tweet_path)
        documents = list(ds[content_key])
        labels = list(ds["label"]) if "label" in ds.column_names else None
        datasets["tweet_topic"] = (documents, labels)
        print(f"  Loaded {len(documents)} documents")
    except Exception as e:
        print(f"  Failed to load tweet_topic: {e}")
    
    return datasets


def compute_dataset_stats(name: str, documents: list[str], labels: Optional[list] = None) -> DatasetStats:
    """Compute statistics for a dataset."""
    doc_lengths_chars = [len(doc) for doc in documents]
    doc_lengths_words = [len(doc.split()) for doc in documents]
    
    label_dist = None
    num_labels = None
    if labels is not None:
        from collections import Counter
        label_counts = Counter(labels)
        num_labels = len(label_counts)
        label_dist = dict(label_counts)
    
    return DatasetStats(
        name=name,
        num_documents=len(documents),
        avg_doc_length_chars=np.mean(doc_lengths_chars),
        avg_doc_length_words=np.mean(doc_lengths_words),
        min_doc_length_chars=min(doc_lengths_chars),
        max_doc_length_chars=max(doc_lengths_chars),
        min_doc_length_words=min(doc_lengths_words),
        max_doc_length_words=max(doc_lengths_words),
        num_labels=num_labels,
        label_distribution=label_dist,
    )


def print_dataset_stats(stats_list: list[DatasetStats]):
    """Print dataset statistics in a formatted table."""
    print("\n" + "="*100)
    print("DATASET STATISTICS")
    print("="*100)
    
    header = f"{'Dataset':<20} {'# Docs':<10} {'Avg Chars':<12} {'Avg Words':<12} {'Min Words':<12} {'Max Words':<12} {'# Labels':<10}"
    print(header)
    print("-"*100)
    
    for stats in stats_list:
        num_labels_str = str(stats.num_labels) if stats.num_labels else "N/A"
        row = f"{stats.name:<20} {stats.num_documents:<10} {stats.avg_doc_length_chars:<12.1f} {stats.avg_doc_length_words:<12.1f} {stats.min_doc_length_words:<12} {stats.max_doc_length_words:<12} {num_labels_str:<10}"
        print(row)
    
    print("="*100)


def benchmark_sentence_transformer(
    model_name: str,
    documents: list[str],
    batch_size: int = 32,
    device: str = "cuda",
    dataset_name: str = "synthetic"
) -> BenchmarkResult:
    """Benchmark Sentence-BERT embedding model."""
    from sentence_transformers import SentenceTransformer
    
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    device_info = get_device_info(device_obj)
    
    print(f"\n{'='*60}")
    print(f"Benchmarking Sentence-BERT: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Device: {device_info}")
    print(f"Documents: {len(documents)}, Batch size: {batch_size}")
    print(f"{'='*60}")
    
    # Load model
    print("Loading model...")
    load_start = time.time()
    model = SentenceTransformer(model_name, device=str(device_obj), trust_remote_code=True)
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s")
    
    model_size = estimate_model_size_mb(model)
    num_params = count_parameters(model)
    print(f"Model size: {model_size:.1f} MB, Parameters: {num_params / 1e6:.1f}M")
    
    # Get tokenizer for sequence length estimation
    tokenizer = model.tokenizer
    max_seq_length = model.max_seq_length
    print(f"Model max sequence length: {max_seq_length}")
    
    # Estimate average sequence length by sampling (compute actual lengths, not padded)
    sample_size = min(100, len(documents))
    sample_docs = documents[:sample_size]
    sample_encodings = tokenizer(
        sample_docs,
        padding=False,  # Don't pad to get actual lengths
        truncation=True,
        max_length=max_seq_length,
        return_tensors=None  # Return list of lists
    )
    # Compute actual average sequence length per document
    seq_lengths = [len(ids) for ids in sample_encodings['input_ids']]
    avg_seq_length = np.mean(seq_lengths)
    min_seq_length = min(seq_lengths)
    max_seq_length_actual = max(seq_lengths)
    print(f"Sequence lengths - Avg: {avg_seq_length:.1f}, Min: {min_seq_length}, Max: {max_seq_length_actual} (sampled from {sample_size} docs)")
    
    # Warmup
    print("Warming up...")
    _ = model.encode(documents[:min(10, len(documents))], batch_size=batch_size, show_progress_bar=False)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    # Benchmark
    print("Running benchmark...")
    
    # Compute actual total tokens for more accurate FLOPs calculation
    # (do this before timing to not include tokenization in timing)
    all_encodings = tokenizer(
        documents,
        padding=False,
        truncation=True,
        max_length=model.max_seq_length,
        return_tensors=None
    )
    total_tokens = sum(len(ids) for ids in all_encodings['input_ids'])
    actual_avg_seq_length = total_tokens / len(documents)
    print(f"Actual average sequence length (all docs): {actual_avg_seq_length:.1f}")
    
    start_time = time.time()
    _ = model.encode(documents, batch_size=batch_size, show_progress_bar=True)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    total_time = time.time() - start_time
    
    gpu_memory = None
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.max_memory_allocated() / 1024**2
    
    # Compute FLOPs using actual average sequence length
    avg_seq_length = actual_avg_seq_length  # Use actual, not sampled
    flops_per_doc = estimate_transformer_flops(num_params, avg_seq_length, is_encoder=True)
    total_flops = flops_per_doc * len(documents)
    tflops_per_second = (total_flops / total_time) / 1e12  # Convert to TFLOPs/s
    
    print(f"FLOPs per doc: {flops_per_doc / 1e9:.2f} GFLOPs")
    print(f"Total FLOPs: {total_flops / 1e12:.2f} TFLOPs")
    print(f"Throughput: {tflops_per_second:.2f} TFLOPs/s")
    
    result = BenchmarkResult(
        model_name=model_name,
        model_type="embedding",
        model_size_mb=model_size,
        num_parameters=num_params,
        num_documents=len(documents),
        total_time_seconds=total_time,
        docs_per_second=len(documents) / total_time,
        time_per_doc_ms=(total_time / len(documents)) * 1000,
        batch_size=batch_size,
        device=str(device_obj),
        dataset_name=dataset_name,
        avg_seq_length=avg_seq_length,
        flops_per_doc=flops_per_doc,
        total_flops=total_flops,
        tflops_per_second=tflops_per_second,
        gpu_name=device_info.get("gpu_name"),
        gpu_memory_mb=gpu_memory,
        notes=f"Sentence-BERT model ({num_params / 1e6:.1f}M params)"
    )
    
    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return result


def benchmark_llm(
    model_name: str,
    documents: list[str],
    batch_size: int = 8,
    device: str = "cuda",
    max_length: int = 512,
    vocab_size_to_extract: int = 2000,
    dataset_name: str = "synthetic"
) -> BenchmarkResult:
    """Benchmark LLM for next-word logit extraction (our approach)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    device_info = get_device_info(device_obj)
    
    print(f"\n{'='*60}")
    print(f"Benchmarking LLM: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Device: {device_info}")
    print(f"Documents: {len(documents)}, Batch size: {batch_size}")
    print(f"{'='*60}")
    
    # Load tokenizer and model
    print("Loading model...")
    load_start = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    
    # Try to use flash attention and bfloat16 for efficiency
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).eval()
    except Exception:
        # Fallback without flash attention
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        ).eval()
    
    model.to(device_obj)
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s")
    
    model_size = estimate_model_size_mb(model)
    num_params = count_parameters(model)
    print(f"Model size: {model_size:.1f} MB, Parameters: {num_params / 1e9:.2f}B")
    
    # Create a simple prompt template
    def create_prompt(doc: str) -> str:
        return f"Document: {doc}\n\nThis document is about"
    
    # Estimate average sequence length by sampling
    sample_size = min(100, len(documents))
    sample_docs = documents[:sample_size]
    sample_prompts = [create_prompt(doc) for doc in sample_docs]
    sample_encodings = tokenizer(
        sample_prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    avg_seq_length = sample_encodings['input_ids'].shape[1]
    print(f"Average sequence length (sampled): {avg_seq_length}")
    
    # Warmup
    print("Warming up...")
    warmup_docs = documents[:min(2, len(documents))]
    warmup_prompts = [create_prompt(doc) for doc in warmup_docs]
    warmup_inputs = tokenizer(
        warmup_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device_obj)
    
    with torch.no_grad():
        _ = model(**warmup_inputs, use_cache=False, output_hidden_states=True)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    # Benchmark
    print("Running benchmark...")
    start_time = time.time()
    
    total_tokens_processed = 0
    all_logits = []
    for i in tqdm(range(0, len(documents), batch_size), desc="Processing batches"):
        batch_docs = documents[i:i + batch_size]
        prompts = [create_prompt(doc) for doc in batch_docs]
        
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device_obj)
        
        # Track actual tokens processed
        total_tokens_processed += inputs['input_ids'].numel()
        
        with torch.no_grad():
            outputs = model(
                **inputs,
                use_cache=False,
                output_hidden_states=True
            )
        
        # Extract next-word logits (simulating our approach)
        logits = outputs.logits[:, -1, :vocab_size_to_extract]
        all_logits.append(logits.cpu())
        
        # Also extract embeddings from hidden states
        hidden_states = outputs.hidden_states[-1]  # Last layer
        # Use last token embedding
        
        del outputs, logits, hidden_states
        
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    total_time = time.time() - start_time
    
    gpu_memory = None
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.max_memory_allocated() / 1024**2
    
    # Compute FLOPs (using actual average tokens per document)
    actual_avg_seq_length = total_tokens_processed / len(documents)
    flops_per_doc = estimate_transformer_flops(num_params, actual_avg_seq_length, is_encoder=False)
    total_flops = flops_per_doc * len(documents)
    tflops_per_second = (total_flops / total_time) / 1e12  # Convert to TFLOPs/s
    
    print(f"Actual avg sequence length: {actual_avg_seq_length:.1f}")
    print(f"FLOPs per doc: {flops_per_doc / 1e9:.2f} GFLOPs")
    print(f"Total FLOPs: {total_flops / 1e12:.2f} TFLOPs")
    print(f"Throughput: {tflops_per_second:.2f} TFLOPs/s")
    
    result = BenchmarkResult(
        model_name=model_name,
        model_type="llm",
        model_size_mb=model_size,
        num_parameters=num_params,
        num_documents=len(documents),
        total_time_seconds=total_time,
        docs_per_second=len(documents) / total_time,
        time_per_doc_ms=(total_time / len(documents)) * 1000,
        batch_size=batch_size,
        device=str(device_obj),
        dataset_name=dataset_name,
        avg_seq_length=actual_avg_seq_length,
        flops_per_doc=flops_per_doc,
        total_flops=total_flops,
        tflops_per_second=tflops_per_second,
        gpu_name=device_info.get("gpu_name"),
        gpu_memory_mb=gpu_memory,
        notes=f"LLM for next-word logit extraction ({num_params / 1e9:.2f}B params)"
    )
    
    # Cleanup
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return result


def print_results_table(results: list[BenchmarkResult], dataset_name: str = None):
    """Print results in a formatted table."""
    # Filter results by dataset if specified
    if dataset_name:
        results = [r for r in results if r.dataset_name == dataset_name]
    
    if not results:
        print("No results to display.")
        return
    
    print("\n" + "="*160)
    if dataset_name:
        print(f"BENCHMARK RESULTS - Dataset: {dataset_name}")
    else:
        print("BENCHMARK RESULTS SUMMARY (All Datasets)")
    print("="*160)
    
    # Header
    header = f"{'Model':<40} {'Dataset':<15} {'Type':<10} {'Params':<12} {'AvgSeqLen':<10} {'Docs/s':<10} {'ms/doc':<10} {'GFLOPs/doc':<12} {'TFLOPs/s':<10}"
    print(header)
    print("-"*160)
    
    # Sort by dataset, model type, and then by speed
    sorted_results = sorted(results, key=lambda x: (x.dataset_name, x.model_type, -x.docs_per_second))
    
    current_type = None
    current_dataset = None
    for r in sorted_results:
        if r.dataset_name != current_dataset:
            if current_dataset is not None:
                print()
            current_dataset = r.dataset_name
            current_type = None
            print(f"\n--- Dataset: {current_dataset} ({r.num_documents} documents) ---")
            print("-"*160)
        
        if r.model_type != current_type:
            current_type = r.model_type
            type_label = "EMBEDDING MODELS (Baselines)" if current_type == "embedding" else "LLM MODELS (Our Approach)"
            print(f"\n  {type_label}")
        
        # Format parameters
        if r.num_parameters >= 1e9:
            params_str = f"{r.num_parameters / 1e9:.2f}B"
        else:
            params_str = f"{r.num_parameters / 1e6:.1f}M"
        
        gflops_per_doc = r.flops_per_doc / 1e9
        
        row = f"  {r.model_name:<38} {r.dataset_name:<15} {r.model_type:<10} {params_str:<12} {r.avg_seq_length:<10.1f} {r.docs_per_second:<10.1f} {r.time_per_doc_ms:<10.2f} {gflops_per_doc:<12.2f} {r.tflops_per_second:<10.2f}"
        print(row)
    
    print("="*160)


def print_overhead_comparison(results: list[BenchmarkResult]):
    """Print overhead comparison between embedding and LLM models."""
    # Group results by dataset
    datasets = set(r.dataset_name for r in results)
    
    print("\n" + "="*140)
    print("OVERHEAD COMPARISON BY DATASET")
    print("="*140)
    
    for dataset_name in sorted(datasets):
        dataset_results = [r for r in results if r.dataset_name == dataset_name]
        embedding_results = [r for r in dataset_results if r.model_type == "embedding"]
        llm_results = [r for r in dataset_results if r.model_type == "llm"]
        
        if not embedding_results or not llm_results:
            continue
        
        print(f"\n--- Dataset: {dataset_name} ---")
        print("-"*100)
        
        # Find the fastest embedding model
        fastest_embedding = min(embedding_results, key=lambda x: x.time_per_doc_ms)
        
        print(f"\nFastest baseline embedding: {fastest_embedding.model_name}")
        print(f"  - {fastest_embedding.time_per_doc_ms:.2f} ms/doc ({fastest_embedding.docs_per_second:.1f} docs/s)")
        print(f"  - {fastest_embedding.flops_per_doc / 1e9:.2f} GFLOPs/doc, {fastest_embedding.tflops_per_second:.2f} TFLOPs/s")
        
        print(f"\nLLM Models (Our Approach):")
        for llm_r in sorted(llm_results, key=lambda x: x.time_per_doc_ms):
            time_overhead_ratio = llm_r.time_per_doc_ms / fastest_embedding.time_per_doc_ms
            flops_overhead_ratio = llm_r.flops_per_doc / fastest_embedding.flops_per_doc
            print(f"  - {llm_r.model_name}")
            print(f"    Time: {llm_r.time_per_doc_ms:.2f} ms/doc ({llm_r.docs_per_second:.1f} docs/s) - {time_overhead_ratio:.2f}x overhead")
            print(f"    FLOPs: {llm_r.flops_per_doc / 1e9:.2f} GFLOPs/doc - {flops_overhead_ratio:.2f}x overhead")
            print(f"    Throughput: {llm_r.tflops_per_second:.2f} TFLOPs/s")
        
        # Compare with commonly used all-mpnet-base-v2 (used in ZeroShotTM)
        mpnet_results = [r for r in embedding_results if "mpnet-base" in r.model_name.lower()]
        if mpnet_results:
            mpnet = mpnet_results[0]
            print(f"\n  Comparison with ZeroShotTM's default (all-mpnet-base-v2):")
            print(f"    {mpnet.model_name}: {mpnet.time_per_doc_ms:.2f} ms/doc, {mpnet.flops_per_doc / 1e9:.2f} GFLOPs/doc")
            for llm_r in sorted(llm_results, key=lambda x: x.time_per_doc_ms):
                time_ratio = llm_r.time_per_doc_ms / mpnet.time_per_doc_ms
                flops_ratio = llm_r.flops_per_doc / mpnet.flops_per_doc
                print(f"    {llm_r.model_name}: {llm_r.time_per_doc_ms:.2f} ms/doc ({time_ratio:.2f}x time), {llm_r.flops_per_doc / 1e9:.2f} GFLOPs ({flops_ratio:.2f}x FLOPs)")
    
    print("\n" + "="*140)


def print_cross_dataset_summary(results: list[BenchmarkResult]):
    """Print summary statistics across all datasets."""
    print("\n" + "="*160)
    print("CROSS-DATASET SUMMARY")
    print("="*160)
    
    # Group by model
    models = set(r.model_name for r in results)
    
    print(f"\n{'Model':<45} {'Params':<12} {'Avg ms/doc':<12} {'Std ms/doc':<12} {'Avg GFLOPs':<12} {'Avg TFLOPs/s':<12} {'Datasets':<10}")
    print("-"*160)
    
    # Embedding models first
    print("\nEMBEDDING MODELS:")
    embedding_models = set(r.model_name for r in results if r.model_type == "embedding")
    for model in sorted(embedding_models):
        model_results = [r for r in results if r.model_name == model]
        avg_time = np.mean([r.time_per_doc_ms for r in model_results])
        std_time = np.std([r.time_per_doc_ms for r in model_results])
        avg_gflops = np.mean([r.flops_per_doc / 1e9 for r in model_results])
        avg_tflops_s = np.mean([r.tflops_per_second for r in model_results])
        num_datasets = len(model_results)
        
        # Format parameters
        num_params = model_results[0].num_parameters
        if num_params >= 1e9:
            params_str = f"{num_params / 1e9:.2f}B"
        else:
            params_str = f"{num_params / 1e6:.1f}M"
        
        print(f"  {model:<43} {params_str:<12} {avg_time:<12.2f} {std_time:<12.2f} {avg_gflops:<12.2f} {avg_tflops_s:<12.2f} {num_datasets:<10}")
    
    # LLM models
    print("\nLLM MODELS:")
    llm_models = set(r.model_name for r in results if r.model_type == "llm")
    for model in sorted(llm_models):
        model_results = [r for r in results if r.model_name == model]
        avg_time = np.mean([r.time_per_doc_ms for r in model_results])
        std_time = np.std([r.time_per_doc_ms for r in model_results])
        avg_gflops = np.mean([r.flops_per_doc / 1e9 for r in model_results])
        avg_tflops_s = np.mean([r.tflops_per_second for r in model_results])
        num_datasets = len(model_results)
        
        # Format parameters
        num_params = model_results[0].num_parameters
        if num_params >= 1e9:
            params_str = f"{num_params / 1e9:.2f}B"
        else:
            params_str = f"{num_params / 1e6:.1f}M"
        
        print(f"  {model:<43} {params_str:<12} {avg_time:<12.2f} {std_time:<12.2f} {avg_gflops:<12.2f} {avg_tflops_s:<12.2f} {num_datasets:<10}")
    
    print("="*160)


def print_flops_summary(results: list[BenchmarkResult]):
    """Print a dedicated FLOPs summary table."""
    print("\n" + "="*140)
    print("FLOPS ANALYSIS SUMMARY")
    print("="*140)
    
    # Get unique models
    models = []
    seen = set()
    for r in results:
        if r.model_name not in seen:
            models.append(r)
            seen.add(r.model_name)
    
    # Sort by FLOPs per doc
    models_sorted = sorted(models, key=lambda x: x.flops_per_doc)
    
    print(f"\n{'Model':<45} {'Type':<10} {'Params':<12} {'Avg SeqLen':<12} {'GFLOPs/doc':<14} {'Relative FLOPs':<15}")
    print("-"*140)
    
    # Use smallest model as baseline
    baseline_flops = models_sorted[0].flops_per_doc
    
    for r in models_sorted:
        # Format parameters
        if r.num_parameters >= 1e9:
            params_str = f"{r.num_parameters / 1e9:.2f}B"
        else:
            params_str = f"{r.num_parameters / 1e6:.1f}M"
        
        gflops = r.flops_per_doc / 1e9
        relative = r.flops_per_doc / baseline_flops
        
        print(f"{r.model_name:<45} {r.model_type:<10} {params_str:<12} {r.avg_seq_length:<12.1f} {gflops:<14.2f} {relative:<15.2f}x")
    
    print("="*140)
    
    # Print key insights
    print("\n KEY INSIGHTS:")
    embedding_models = [r for r in models if r.model_type == "embedding"]
    llm_models = [r for r in models if r.model_type == "llm"]
    
    if embedding_models and llm_models:
        min_embedding_flops = min(r.flops_per_doc for r in embedding_models)
        max_embedding_flops = max(r.flops_per_doc for r in embedding_models)
        min_llm_flops = min(r.flops_per_doc for r in llm_models)
        max_llm_flops = max(r.flops_per_doc for r in llm_models)
        
        print(f"  - Embedding models FLOPs range: {min_embedding_flops/1e9:.2f} - {max_embedding_flops/1e9:.2f} GFLOPs/doc")
        print(f"  - LLM models FLOPs range: {min_llm_flops/1e9:.2f} - {max_llm_flops/1e9:.2f} GFLOPs/doc")
        print(f"  - LLM/Embedding FLOPs ratio range: {min_llm_flops/max_embedding_flops:.2f}x - {max_llm_flops/min_embedding_flops:.2f}x")
    
    print("="*140)


def save_results(results: list[BenchmarkResult], dataset_stats: list[DatasetStats], output_path: str):
    """Save results to JSON file."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "dataset_stats": [asdict(s) for s in dataset_stats],
        "results": [asdict(r) for r in results]
    }
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark embedding overhead for topic model preprocessing"
    )
    parser.add_argument(
        "--num_docs", type=int, default=None,
        help="Number of documents to benchmark (None = use all available)"
    )
    parser.add_argument(
        "--batch_size_embedding", type=int, default=32,
        help="Batch size for embedding models"
    )
    parser.add_argument(
        "--batch_size_llm", type=int, default=8,
        help="Batch size for LLM models"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--output", type=str, default="analysis/benchmark_results.json",
        help="Output path for results JSON"
    )
    parser.add_argument(
        "--skip_llm", action="store_true",
        help="Skip LLM benchmarks (only run embedding models)"
    )
    parser.add_argument(
        "--skip_embedding", action="store_true",
        help="Skip embedding benchmarks (only run LLM models)"
    )
    parser.add_argument(
        "--use_synthetic", action="store_true",
        help="Use synthetic data instead of real datasets"
    )
    parser.add_argument(
        "--synthetic_num_docs", type=int, default=1000,
        help="Number of synthetic documents (only used with --use_synthetic)"
    )
    parser.add_argument(
        "--embedding_models", type=str, nargs="+",
        default=[
            "all-mpnet-base-v2",           # Used by ZeroShotTM/CombinedTM (109M params)
            "all-MiniLM-L6-v2",            # Used by BERTopic default (22M params)
            "paraphrase-MiniLM-L6-v2",     # Alternative lightweight option
            "all-MiniLM-L12-v2",           # Larger MiniLM variant
            "Alibaba-NLP/gte-large-en-v1.5",  # Large GTE model (434M params)
        ],
        help="Sentence-BERT models to benchmark"
    )
    parser.add_argument(
        "--llm_models", type=str, nargs="+",
        default=[
            "baidu/ERNIE-4.5-0.3B-PT",           # Our smallest model (0.3B)
            "meta-llama/Llama-3.2-1B-Instruct",  # Llama 1B
            "meta-llama/Llama-3.1-8B-Instruct",  # Llama 8B
        ],
        help="LLM models to benchmark"
    )
    parser.add_argument(
        "--content_key", type=str, default="text",
        help="Key for document content in datasets"
    )
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    print(f"\n{'#'*80}")
    print(f"# EMBEDDING OVERHEAD BENCHMARK")
    print(f"# Date: {datetime.now().isoformat()}")
    print(f"# Device: {args.device}")
    if args.device == "cuda":
        print(f"# GPU: {torch.cuda.get_device_name(0)}")
        print(f"# GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"{'#'*80}")
    
    # Load datasets
    all_dataset_stats = []
    if args.use_synthetic:
        print("\nUsing synthetic documents...")
        documents = generate_synthetic_documents(args.synthetic_num_docs)
        datasets = {"synthetic": (documents, None)}
        stats = compute_dataset_stats("synthetic", documents)
        all_dataset_stats.append(stats)
    else:
        print("\nLoading real datasets...")
        datasets = load_real_datasets(args.content_key)
        
        for name, (docs, labels) in datasets.items():
            stats = compute_dataset_stats(name, docs, labels)
            all_dataset_stats.append(stats)
    
    # Print dataset statistics
    print_dataset_stats(all_dataset_stats)
    
    # Limit documents if specified
    if args.num_docs is not None:
        print(f"\nLimiting to {args.num_docs} documents per dataset")
        datasets = {
            name: (docs[:args.num_docs], labels[:args.num_docs] if labels else None)
            for name, (docs, labels) in datasets.items()
        }
    
    results = []
    
    # Run benchmarks for each dataset
    for dataset_name, (documents, labels) in datasets.items():
        print(f"\n{'#'*80}")
        print(f"# BENCHMARKING ON DATASET: {dataset_name} ({len(documents)} documents)")
        print(f"{'#'*80}")
        
        # Benchmark embedding models (baselines)
        if not args.skip_embedding:
            print("\n" + "#"*60)
            print("# BENCHMARKING EMBEDDING MODELS (Used in Baselines)")
            print("#"*60)
            
            for model_name in args.embedding_models:
                try:
                    result = benchmark_sentence_transformer(
                        model_name=model_name,
                        documents=documents,
                        batch_size=args.batch_size_embedding,
                        device=args.device,
                        dataset_name=dataset_name
                    )
                    results.append(result)
                    print(f"✓ {model_name}: {result.docs_per_second:.1f} docs/s, {result.flops_per_doc/1e9:.2f} GFLOPs/doc")
                except Exception as e:
                    print(f"✗ {model_name}: Failed - {e}")
                    import traceback
                    traceback.print_exc()
        
        # Benchmark LLM models (our approach)
        if not args.skip_llm:
            print("\n" + "#"*60)
            print("# BENCHMARKING LLM MODELS (Our Approach)")
            print("#"*60)
            
            for model_name in args.llm_models:
                try:
                    result = benchmark_llm(
                        model_name=model_name,
                        documents=documents,
                        batch_size=args.batch_size_llm,
                        device=args.device,
                        dataset_name=dataset_name
                    )
                    results.append(result)
                    print(f"✓ {model_name}: {result.docs_per_second:.1f} docs/s, {result.flops_per_doc/1e9:.2f} GFLOPs/doc")
                except Exception as e:
                    print(f"✗ {model_name}: Failed - {e}")
                    import traceback
                    traceback.print_exc()
    
    # Print and save results
    if results:
        print_results_table(results)
        print_flops_summary(results)
        print_overhead_comparison(results)
        print_cross_dataset_summary(results)
        save_results(results, all_dataset_stats, args.output)
    else:
        print("\nNo successful benchmarks to report.")


if __name__ == "__main__":
    main()
