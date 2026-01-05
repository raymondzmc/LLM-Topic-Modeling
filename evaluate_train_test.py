"""Train topic model on train set and evaluate clustering metrics on test set."""

import os
import json
import argparse
import random
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from data.loaders import load_training_data
from data.dataset.ctm_dataset import get_ctm_dataset_from_processed_data
from models.ctm import GenerativeTM


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def calculate_purity(true_labels, pred_labels):
    """
    Calculate harmonic purity between two set of clusterings.

    Parameters:
    - true_labels: Ground-truth labels for each document
    - pred_labels: Predicted labels for each document (topic assignments)

    Returns:
    - purity: Purity score
    - inverse_purity: Inverse purity score
    - harmonic_purity: Harmonic purity score
    """
    # Create a DataFrame for contingency matrix calculation
    df = pd.DataFrame({'true': true_labels, 'pred': pred_labels})
    
    contingency_matrix = metrics.cluster.contingency_matrix(df['true'], df['pred'])
    precision = contingency_matrix / contingency_matrix.sum(axis=0).reshape(1, -1)
    recall = contingency_matrix / contingency_matrix.sum(axis=1).reshape(-1, 1)
    f1 = 2 * (precision * recall) / (precision + recall)
    f1 = np.nan_to_num(f1)
    purity = (
        np.amax(precision, axis=0) * contingency_matrix.sum(axis=0)
    ).sum() / contingency_matrix.sum()
    inverse_purity = (
        np.amax(recall, axis=1) * contingency_matrix.sum(axis=1)
    ).sum() / contingency_matrix.sum()
    harmonic_purity = (
        np.amax(f1, axis=1) * contingency_matrix.sum(axis=1)
    ).sum() / contingency_matrix.sum()
    return purity, inverse_purity, harmonic_purity


def run(args: argparse.Namespace):
    """Main training and evaluation loop."""
    print(f"\n{'='*60}")
    print(f"Train/Test Topic Model Evaluation")
    print(f"{'='*60}")
    print(f"Train data: {args.train_data_path}")
    print(f"Test data: {args.test_data_path}")
    print(f"Num topics: {args.num_topics}")
    print(f"Num seeds: {args.num_seeds}")
    print(f"{'='*60}\n")
    
    # Load training data
    print("Loading training data...")
    train_data = load_training_data(args.train_data_path, for_generative=True)
    
    # Load test data
    print("Loading test data...")
    test_data = load_training_data(args.test_data_path, for_generative=True)
    
    # Create CTM datasets
    print("Creating CTM datasets...")
    train_ctm_dataset = get_ctm_dataset_from_processed_data(
        train_data.processed_dataset,
        train_data.vocab,
    )
    
    test_ctm_dataset = get_ctm_dataset_from_processed_data(
        test_data.processed_dataset,
        test_data.vocab,  # Should be same as train vocab
    )
    
    # Get test labels
    test_labels = test_data.labels
    if test_labels is None:
        raise ValueError("Test dataset does not have labels for evaluation")
    test_labels = np.array(test_labels)
    
    # Store results for each seed
    all_results = []
    
    for seed in range(args.num_seeds):
        set_seed(seed)
        print(f"\n[Seed {seed}] Training GenerativeTM...")
        
        # Train model on training data
        model = GenerativeTM(
            vocab_size=len(train_data.vocab),
            embedding_size=train_ctm_dataset.x_embeddings.shape[1],
            num_topics=args.num_topics,
            activation=args.activation,
            hidden_sizes=tuple([args.hidden_size] * args.num_hidden_layers),
            solver=args.solver,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            loss_weight=args.loss_weight,
            sparsity_ratio=args.sparsity_ratio,
            loss_type=args.loss_type,
            temperature=args.temperature,
            top_words=args.top_words,
        )
        model.fit(train_ctm_dataset)
        
        # Get topic assignments for test set
        print(f"[Seed {seed}] Getting topic assignments for test set...")
        test_thetas = model.get_thetas(test_ctm_dataset)  # Shape: (n_docs, n_topics)
        pred_labels = np.argmax(test_thetas, axis=1)
        
        # Compute clustering metrics
        print(f"[Seed {seed}] Computing clustering metrics...")
        
        # Purity metrics
        purity, inverse_purity, harmonic_purity = calculate_purity(test_labels, pred_labels)
        
        # ARI and NMI
        ari = adjusted_rand_score(test_labels, pred_labels)
        nmi = normalized_mutual_info_score(test_labels, pred_labels)
        
        seed_results = {
            'seed': seed,
            'purity': purity,
            'inverse_purity': inverse_purity,
            'harmonic_purity': harmonic_purity,
            'ari': ari,
            'nmi': nmi,
        }
        all_results.append(seed_results)
        
        print(f"[Seed {seed}] Results:")
        print(f"  Purity: {purity:.4f}")
        print(f"  Inverse Purity: {inverse_purity:.4f}")
        print(f"  Harmonic Purity: {harmonic_purity:.4f}")
        print(f"  ARI: {ari:.4f}")
        print(f"  NMI: {nmi:.4f}")
    
    # Compute averaged results
    print(f"\n{'='*60}")
    print("Averaged Results (mean ± std):")
    print(f"{'='*60}")
    
    metrics_names = ['purity', 'inverse_purity', 'harmonic_purity', 'ari', 'nmi']
    avg_results = {}
    for metric in metrics_names:
        values = [r[metric] for r in all_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        avg_results[f'{metric}_mean'] = mean_val
        avg_results[f'{metric}_std'] = std_val
        print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Save results if output path provided
    if args.output_path:
        output = {
            'config': {
                'train_data_path': args.train_data_path,
                'test_data_path': args.test_data_path,
                'num_topics': args.num_topics,
                'num_seeds': args.num_seeds,
            },
            'per_seed_results': all_results,
            'averaged_results': avg_results,
        }
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.output_path}")
    
    return avg_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train topic model on train set and evaluate on test set")
    
    # Data arguments
    parser.add_argument('--train_data_path', type=str, required=True,
                        help='Path to processed train data directory or HF repo ID')
    parser.add_argument('--test_data_path', type=str, required=True,
                        help='Path to processed test data directory or HF repo ID')
    
    # Model arguments
    parser.add_argument('--num_topics', type=int, default=22, help='Number of topics')
    parser.add_argument('--top_words', type=int, default=15, help='Top words per topic')
    
    # Training arguments
    parser.add_argument('--num_seeds', type=int, default=5, help='Number of random seeds')
    parser.add_argument('--num_epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=200, help='Hidden layer size')
    parser.add_argument('--num_hidden_layers', type=int, default=2, help='Hidden layers')
    parser.add_argument('--activation', type=str, default='softplus', help='Activation')
    parser.add_argument('--solver', type=str, default='adam', help='Optimizer')
    
    # Generative model arguments
    parser.add_argument('--loss_weight', type=float, default=1e3, help='Reconstruction loss weight')
    parser.add_argument('--sparsity_ratio', type=float, default=1.0, help='Sparsity ratio')
    parser.add_argument('--loss_type', type=str, default='KL', choices=['KL', 'CE'], help='Loss type')
    parser.add_argument('--temperature', type=float, default=3.0, help='Softmax temperature')
    
    # Output arguments
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save results JSON')
    
    args = parser.parse_args()
    run(args)

