"""Retrieval-based evaluation for baseline representations (BoW and LLM targets).

This script evaluates retrieval performance using:
1. BoW (Bag-of-Words): Term-frequency vectors
2. LLM Targets: Next-token probability distributions from the LLM

Data is loaded directly from HuggingFace Hub.
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
from collections import Counter, defaultdict

from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax

from data.loaders import load_or_download_dataset
from evaluate_retrieval import (
    compute_pairwise_kl_divergence_torch,
    compute_precision_at_k,
    apply_subsetting,
)


def compute_pairwise_cosine_similarity(X: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix.
    
    Args:
        X: Document representations (n_docs, n_features)
        
    Returns:
        np.ndarray: Pairwise cosine similarity matrix (n_docs, n_docs)
    """
    return cosine_similarity(X)


def bow_to_term_frequency(bow_strings: list, vocab: list) -> np.ndarray:
    """Convert BoW strings to term-frequency vectors.
    
    Args:
        bow_strings: List of space-separated token strings
        vocab: Vocabulary list
        
    Returns:
        np.ndarray of shape (n_docs, vocab_size) with term frequencies
    """
    vocab2idx = {word: i for i, word in enumerate(vocab)}
    n_docs = len(bow_strings)
    vocab_size = len(vocab)
    
    tf_matrix = np.zeros((n_docs, vocab_size), dtype=np.float32)
    
    for i, bow_str in enumerate(tqdm(bow_strings, desc="Converting BoW to TF")):
        tokens = bow_str.split()
        token_counts = Counter(tokens)
        for token, count in token_counts.items():
            if token in vocab2idx:
                tf_matrix[i, vocab2idx[token]] = count
    
    return tf_matrix


def load_llm_targets(dataset) -> np.ndarray:
    """Load LLM next-token logits from dataset and convert to probabilities.
    
    Args:
        dataset: HuggingFace dataset with 'next_word_logits' column
        
    Returns:
        np.ndarray of shape (n_docs, vocab_size) with softmax probabilities
    """
    n_docs = len(dataset)
    first_logits = np.array(dataset[0]['next_word_logits'])
    logits_dim = first_logits.shape[0]
    
    logits_matrix = np.zeros((n_docs, logits_dim), dtype=np.float32)
    
    for i, item in enumerate(tqdm(dataset, desc="Loading LLM targets")):
        logits_matrix[i] = np.array(item['next_word_logits'])
    
    # Apply softmax to convert logits to probability distributions
    print("Applying softmax to convert logits to probabilities...")
    prob_matrix = softmax(logits_matrix, axis=1)
    
    return prob_matrix


def evaluate_retrieval_baseline(
    representation: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    baseline_type: str,
    subset_size: int = None,
) -> dict:
    """Evaluate retrieval using a given representation.
    
    Args:
        representation: Document representations (n_docs, n_features)
        labels: Document labels
        device: Torch device
        baseline_type: 'bow' (uses cosine similarity) or 'llm_targets' (uses KL divergence)
        subset_size: Optional subsetting for balanced evaluation
        
    Returns:
        Dictionary with precision@k metrics
    """
    # Apply subsetting if specified
    if subset_size is not None:
        labels, representation = apply_subsetting(labels, representation, subset_size)
    
    n_docs = len(labels)
    print(f"Evaluating retrieval on {n_docs} documents...")
    
    # Compute similarity/distance matrix based on baseline type
    if baseline_type == 'bow':
        # BoW: Use cosine similarity (higher = more similar)
        print("Computing pairwise cosine similarity...")
        similarity_matrix = compute_pairwise_cosine_similarity(representation)
        use_similarity = True  # Higher is better
    else:
        # LLM targets: Use KL divergence (lower = more similar)
        print("Computing pairwise KL divergence...")
        representation_tensor = torch.tensor(representation, dtype=torch.float32)
        similarity_matrix = compute_pairwise_kl_divergence_torch(representation_tensor, device).numpy()
        use_similarity = False  # Lower is better (it's a distance)
    
    # For each document, rank others by similarity/distance
    print("Computing precision@k...")
    all_results = defaultdict(list)
    
    for i in tqdm(range(n_docs), desc="Precision@k"):
        row = similarity_matrix[i].copy()
        
        if use_similarity:
            # Exclude self (set to -infinity for similarity)
            row[i] = float('-inf')
            # Sort by similarity (descending = most similar first)
            retrieved_indices = np.argsort(row)[::-1]
        else:
            # Exclude self (set to infinity for distance)
            row[i] = float('inf')
            # Sort by distance (ascending = most similar first)
            retrieved_indices = np.argsort(row)
        
        # Compute precision@k
        results = compute_precision_at_k(retrieved_indices, labels[i], labels)
        for k, v in results.items():
            all_results[k].append(v)
    
    # Average across all documents
    avg_results = {k: np.mean(v) for k, v in all_results.items()}
    
    return avg_results


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset from HuggingFace
    print(f"\nLoading dataset from: {args.data_path}")
    dataset, vocab, metadata = load_or_download_dataset(args.data_path)
    
    # Get labels
    if 'label' not in dataset.column_names:
        raise ValueError("Dataset does not have 'label' column")
    labels = np.array(dataset['label'])
    
    print(f"Dataset: {len(dataset)} documents, {len(vocab)} vocab, {len(set(labels))} labels")
    
    # Load representation based on baseline type
    if args.baseline == 'bow':
        print("\n--- BoW Baseline ---")
        bow_strings = dataset['bow']
        representation = bow_to_term_frequency(bow_strings, vocab)
    elif args.baseline == 'llm_targets':
        print("\n--- LLM Targets Baseline ---")
        representation = load_llm_targets(dataset)
    else:
        raise ValueError(f"Unknown baseline: {args.baseline}")
    
    print(f"Representation shape: {representation.shape}")
    
    # Evaluate retrieval
    results = evaluate_retrieval_baseline(
        representation=representation,
        labels=labels,
        device=device,
        baseline_type=args.baseline,
        subset_size=args.subset_size,
    )
    
    # Print results
    print("\n" + "=" * 50)
    print(f"RESULTS: {args.baseline} baseline")
    print("=" * 50)
    for k, v in sorted(results.items()):
        print(f"  {k}: {v:.4f}")
    print("=" * 50)
    
    # Log to W&B if project specified
    if args.wandb_project:
        import wandb
        from settings import settings
        
        # Extract dataset name from data_path
        dataset_name = args.data_path.split('/')[-1]
        
        run = wandb.init(
            project=args.wandb_project,
            entity=settings.wandb_entity,
            name=f"baseline_{args.baseline}_{dataset_name}",
            config={
                "baseline": args.baseline,
                "data_path": args.data_path,
                "dataset_name": dataset_name,
                "n_docs": len(dataset),
                "vocab_size": len(vocab),
                "n_labels": len(set(labels)),
                "subset_size": args.subset_size,
            },
            mode='online' if not args.wandb_offline else 'offline',
        )
        
        # Log results
        run.log({f"retrieval/{k}": v for k, v in results.items()})
        run.finish()
        
        print(f"\nLogged to W&B: {settings.wandb_entity}/{args.wandb_project}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval baselines (BoW, LLM targets)")
    
    parser.add_argument("--data_path", type=str, required=True,
                        help="HuggingFace repo ID (e.g., raymondzmc/20_newsgroups_ERNIE-4.5-0.3B-PT_vocab_2000_last)")
    parser.add_argument("--baseline", type=str, required=True, choices=['bow', 'llm_targets'],
                        help="Baseline type: 'bow' or 'llm_targets'")
    parser.add_argument("--subset_size", type=int, default=None,
                        help="Optional: subset documents for balanced evaluation")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="W&B project name for logging results")
    parser.add_argument("--wandb_offline", action='store_true',
                        help="Run W&B in offline mode")
    
    args = parser.parse_args()
    main(args)

