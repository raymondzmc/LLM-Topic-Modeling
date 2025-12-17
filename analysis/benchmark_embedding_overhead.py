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
from dataclasses import dataclass, asdict
import warnings

import numpy as np
import torch
from tqdm import tqdm

# Suppress some warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    model_name: str
    model_type: str  # 'embedding' or 'llm'
    model_size_mb: float
    num_documents: int
    total_time_seconds: float
    docs_per_second: float
    time_per_doc_ms: float
    batch_size: int
    device: str
    gpu_name: Optional[str] = None
    gpu_memory_mb: Optional[float] = None
    notes: str = ""


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


def benchmark_sentence_transformer(
    model_name: str,
    documents: list[str],
    batch_size: int = 32,
    device: str = "cuda"
) -> BenchmarkResult:
    """Benchmark Sentence-BERT embedding model."""
    from sentence_transformers import SentenceTransformer
    
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    device_info = get_device_info(device_obj)
    
    print(f"\n{'='*60}")
    print(f"Benchmarking Sentence-BERT: {model_name}")
    print(f"Device: {device_info}")
    print(f"Documents: {len(documents)}, Batch size: {batch_size}")
    print(f"{'='*60}")
    
    # Load model
    print("Loading model...")
    load_start = time.time()
    model = SentenceTransformer(model_name, device=str(device_obj))
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s")
    
    model_size = estimate_model_size_mb(model)
    
    # Warmup
    print("Warming up...")
    _ = model.encode(documents[:min(10, len(documents))], batch_size=batch_size, show_progress_bar=False)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    # Benchmark
    print("Running benchmark...")
    start_time = time.time()
    _ = model.encode(documents, batch_size=batch_size, show_progress_bar=True)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    total_time = time.time() - start_time
    
    gpu_memory = None
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.max_memory_allocated() / 1024**2
    
    result = BenchmarkResult(
        model_name=model_name,
        model_type="embedding",
        model_size_mb=model_size,
        num_documents=len(documents),
        total_time_seconds=total_time,
        docs_per_second=len(documents) / total_time,
        time_per_doc_ms=(total_time / len(documents)) * 1000,
        batch_size=batch_size,
        device=str(device_obj),
        gpu_name=device_info.get("gpu_name"),
        gpu_memory_mb=gpu_memory,
        notes=f"Sentence-BERT model used in baseline topic models"
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
) -> BenchmarkResult:
    """Benchmark LLM for next-word logit extraction (our approach)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    device_info = get_device_info(device_obj)
    
    print(f"\n{'='*60}")
    print(f"Benchmarking LLM: {model_name}")
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
    
    # Create a simple prompt template
    def create_prompt(doc: str) -> str:
        return f"Document: {doc}\n\nThis document is about"
    
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
    
    result = BenchmarkResult(
        model_name=model_name,
        model_type="llm",
        model_size_mb=model_size,
        num_documents=len(documents),
        total_time_seconds=total_time,
        docs_per_second=len(documents) / total_time,
        time_per_doc_ms=(total_time / len(documents)) * 1000,
        batch_size=batch_size,
        device=str(device_obj),
        gpu_name=device_info.get("gpu_name"),
        gpu_memory_mb=gpu_memory,
        notes=f"LLM for next-word logit extraction (our approach)"
    )
    
    # Cleanup
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return result


def print_results_table(results: list[BenchmarkResult]):
    """Print results in a formatted table."""
    print("\n" + "="*100)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*100)
    
    # Header
    header = f"{'Model':<45} {'Type':<10} {'Size(MB)':<10} {'Time(s)':<10} {'Docs/s':<10} {'ms/doc':<10}"
    print(header)
    print("-"*100)
    
    # Sort by model type and then by speed
    sorted_results = sorted(results, key=lambda x: (x.model_type, -x.docs_per_second))
    
    current_type = None
    for r in sorted_results:
        if r.model_type != current_type:
            if current_type is not None:
                print("-"*100)
            current_type = r.model_type
            type_label = "EMBEDDING MODELS (Baselines)" if current_type == "embedding" else "LLM MODELS (Our Approach)"
            print(f"\n{type_label}")
            print("-"*100)
        
        row = f"{r.model_name:<45} {r.model_type:<10} {r.model_size_mb:<10.1f} {r.total_time_seconds:<10.2f} {r.docs_per_second:<10.1f} {r.time_per_doc_ms:<10.2f}"
        print(row)
    
    print("="*100)
    
    # Compute speedup comparison
    embedding_results = [r for r in results if r.model_type == "embedding"]
    llm_results = [r for r in results if r.model_type == "llm"]
    
    if embedding_results and llm_results:
        print("\n" + "="*100)
        print("OVERHEAD COMPARISON")
        print("="*100)
        
        # Find the fastest embedding model (typically used baseline)
        fastest_embedding = min(embedding_results, key=lambda x: x.time_per_doc_ms)
        # Find the smallest/fastest LLM (ERNIE-0.3B)
        fastest_llm = min(llm_results, key=lambda x: x.time_per_doc_ms)
        
        print(f"\nFastest baseline embedding: {fastest_embedding.model_name}")
        print(f"  - {fastest_embedding.time_per_doc_ms:.2f} ms/doc ({fastest_embedding.docs_per_second:.1f} docs/s)")
        
        print(f"\nFastest LLM (Our approach): {fastest_llm.model_name}")
        print(f"  - {fastest_llm.time_per_doc_ms:.2f} ms/doc ({fastest_llm.docs_per_second:.1f} docs/s)")
        
        overhead_ratio = fastest_llm.time_per_doc_ms / fastest_embedding.time_per_doc_ms
        print(f"\nOverhead ratio (LLM / Embedding): {overhead_ratio:.2f}x")
        
        # Compare with commonly used all-mpnet-base-v2 (used in ZeroShotTM)
        mpnet_results = [r for r in embedding_results if "mpnet-base" in r.model_name.lower()]
        if mpnet_results:
            mpnet = mpnet_results[0]
            print(f"\n\nComparison with ZeroShotTM's default (all-mpnet-base-v2):")
            print(f"  - {mpnet.model_name}: {mpnet.time_per_doc_ms:.2f} ms/doc")
            for llm_r in llm_results:
                ratio = llm_r.time_per_doc_ms / mpnet.time_per_doc_ms
                print(f"  - {llm_r.model_name}: {llm_r.time_per_doc_ms:.2f} ms/doc ({ratio:.2f}x overhead)")
        
        print("="*100)


def save_results(results: list[BenchmarkResult], output_path: str):
    """Save results to JSON file."""
    output = {
        "timestamp": datetime.now().isoformat(),
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
        "--num_docs", type=int, default=1000,
        help="Number of synthetic documents to benchmark"
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
        "--embedding_models", type=str, nargs="+",
        default=[
            "all-mpnet-base-v2",      # Used by ZeroShotTM/CombinedTM (109M params)
            "all-MiniLM-L6-v2",       # Used by BERTopic default (22M params)
            "paraphrase-MiniLM-L6-v2",  # Alternative lightweight option
            "all-MiniLM-L12-v2",      # Larger MiniLM variant
        ],
        help="Sentence-BERT models to benchmark"
    )
    parser.add_argument(
        "--llm_models", type=str, nargs="+",
        default=[
            "baidu/ERNIE-4.5-0.3B-PT",        # Our smallest model (0.3B)
            # "meta-llama/Llama-3.2-1B-Instruct",  # Medium model (1B) - uncomment if available
            # "meta-llama/Llama-3.1-8B-Instruct",  # Larger model (8B) - uncomment if available
        ],
        help="LLM models to benchmark"
    )
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    print(f"\n{'#'*60}")
    print(f"# EMBEDDING OVERHEAD BENCHMARK")
    print(f"# Date: {datetime.now().isoformat()}")
    print(f"# Documents: {args.num_docs}")
    print(f"# Device: {args.device}")
    if args.device == "cuda":
        print(f"# GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'#'*60}")
    
    # Generate synthetic documents
    print("\nGenerating synthetic documents...")
    documents = generate_synthetic_documents(args.num_docs)
    print(f"Generated {len(documents)} documents")
    print(f"Sample document: {documents[0][:100]}...")
    
    results = []
    
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
                    device=args.device
                )
                results.append(result)
                print(f"✓ {model_name}: {result.docs_per_second:.1f} docs/s")
            except Exception as e:
                print(f"✗ {model_name}: Failed - {e}")
    
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
                    device=args.device
                )
                results.append(result)
                print(f"✓ {model_name}: {result.docs_per_second:.1f} docs/s")
            except Exception as e:
                print(f"✗ {model_name}: Failed - {e}")
                import traceback
                traceback.print_exc()
    
    # Print and save results
    if results:
        print_results_table(results)
        save_results(results, args.output)
    else:
        print("\nNo successful benchmarks to report.")


if __name__ == "__main__":
    main()

