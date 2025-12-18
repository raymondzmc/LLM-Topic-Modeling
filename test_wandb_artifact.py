#!/usr/bin/env python3
"""Test script to verify W&B artifact uploading works correctly.

This script creates mock model outputs and tests the artifact upload flow
without running actual topic model training.

Usage:
    python test_wandb_artifact.py
"""

import os
import json
import tempfile
import numpy as np
import torch
import wandb

from settings import settings


def create_mock_model_output(num_topics: int = 5, vocab_size: int = 100, num_docs: int = 50, top_words: int = 10):
    """Create mock model output similar to what topic models produce."""
    # Generate mock topic words
    vocab = [f"word_{i}" for i in range(vocab_size)]
    topics = [
        [vocab[j] for j in np.random.choice(vocab_size, top_words, replace=False)]
        for _ in range(num_topics)
    ]
    
    # Generate mock matrices
    topic_word_matrix = np.random.dirichlet(np.ones(vocab_size), size=num_topics)
    topic_doc_matrix = np.random.dirichlet(np.ones(num_topics), size=num_docs).T
    
    return {
        'topics': topics,
        'topic-word-matrix': topic_word_matrix,
        'topic-document-matrix': topic_doc_matrix,
        'training_time': np.random.uniform(10, 100),
    }


def create_mock_evaluation_results():
    """Create mock evaluation results."""
    return {
        'npmi': np.random.uniform(0.05, 0.15),
        'cv': np.random.uniform(0.3, 0.6),
        'diversity': np.random.uniform(0.7, 0.95),
        'training_time': np.random.uniform(10, 100),
    }


def test_artifact_upload():
    """Test the artifact upload flow with mock data."""
    print("=" * 60)
    print("Testing W&B Artifact Upload")
    print("=" * 60)
    
    # Configuration
    model_name = "test_model"
    dataset_name = "test_dataset"
    num_topics = 5
    num_seeds = 2
    top_words = 10
    
    # Create temporary results directory
    with tempfile.TemporaryDirectory() as results_path:
        print(f"\n1. Creating mock results in: {results_path}")
        
        # Create mock outputs for each seed
        for seed in range(num_seeds):
            seed_dir = os.path.join(results_path, f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)
            
            # Create mock model output
            model_output = create_mock_model_output(num_topics, top_words=top_words)
            torch.save(model_output, os.path.join(seed_dir, 'model_output.pt'))
            
            # Save topics as JSON
            with open(os.path.join(seed_dir, 'topics.json'), 'w') as f:
                json.dump(model_output['topics'], f, indent=2)
            
            # Save evaluation results
            eval_results = create_mock_evaluation_results()
            with open(os.path.join(seed_dir, 'evaluation_results.json'), 'w') as f:
                json.dump(eval_results, f, indent=2)
            
            print(f"   Created seed_{seed}/ with model_output.pt, topics.json, evaluation_results.json")
        
        # Create averaged results
        averaged_results = {
            'npmi_mean': 0.1,
            'npmi_std': 0.02,
            'cv_mean': 0.45,
            'cv_std': 0.05,
            'diversity_mean': 0.85,
            'diversity_std': 0.03,
        }
        with open(os.path.join(results_path, 'averaged_results.json'), 'w') as f:
            json.dump(averaged_results, f, indent=2)
        print("   Created averaged_results.json")
        
        # Calculate total size
        total_size_mb = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, _, filenames in os.walk(results_path)
            for filename in filenames
        ) / (1024 * 1024)
        print(f"\n   Total size: {total_size_mb:.4f} MB")
        
        # Initialize W&B run
        print("\n2. Initializing W&B run...")
        run = wandb.init(
            project="test-artifact-upload",
            entity=settings.wandb_entity,
            name=f"{model_name}_K{num_topics}_test",
            config={
                "model": model_name,
                "dataset": dataset_name,
                "num_topics": num_topics,
                "num_seeds": num_seeds,
            },
            mode="online",  # Use "offline" to test without network
        )
        print(f"   Run ID: {run.id}")
        print(f"   Run URL: {run.url}")
        
        # Log some metrics
        print("\n3. Logging metrics...")
        run.log({"test/npmi": 0.1, "test/diversity": 0.85})
        print("   Logged test metrics")
        
        # Create and log artifact
        print("\n4. Creating artifact...")
        artifact_name = f"{model_name}-K{num_topics}-{dataset_name}"
        artifact_description = f"""Topic model outputs for {model_name} on {dataset_name} dataset.

Configuration:
- Number of topics: {num_topics}
- Top words per topic: {top_words}
- Number of seeds: {num_seeds}

Contents per seed:
- model_output.pt: Full model output (topics, topic-word matrix, doc-topic matrix)
- topics.json: Topic word lists for easy viewing
- evaluation_results.json: Coherence, diversity, and clustering metrics

Use averaged_results.json for aggregated metrics across seeds."""

        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            description=artifact_description,
            metadata={
                "model": model_name,
                "dataset": dataset_name,
                "num_topics": num_topics,
                "num_seeds": num_seeds,
                "top_words": top_words,
            }
        )
        
        # Add entire results directory
        artifact.add_dir(results_path)
        print(f"   Added directory: {results_path}")
        
        # Log artifact (asynchronous)
        print("\n5. Logging artifact (async)...")
        run.log_artifact(artifact)
        print(f"   Artifact '{artifact_name}' queued for upload")
        
        # Finish run - this completes pending uploads
        print("\n6. Finishing run (this completes artifact upload)...")
        run.finish()
        print("   Run finished successfully!")
        
        print("\n" + "=" * 60)
        print("Test completed successfully!")
        print("=" * 60)
        print(f"\nView artifacts at: https://wandb.ai/{settings.wandb_entity}/test-artifact-upload/artifacts")


if __name__ == "__main__":
    test_artifact_upload()

