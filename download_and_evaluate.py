#!/usr/bin/env python
"""Download model artifacts from wandb and re-evaluate them."""

import argparse
import json
import os
import torch
import wandb

from settings import settings
from utils.metrics import evaluate_topic_model
from utils.embeddings import get_openai_embedding
from data.loaders import load_training_data


def download_artifact(run_path: str, artifact_name: str, output_dir: str) -> str:
    """Download an artifact from wandb.
    
    Args:
        run_path: Full wandb run path (entity/project/run_id)
        artifact_name: Name of the artifact to download
        output_dir: Directory to save the artifact
        
    Returns:
        Path to the downloaded artifact directory
    """
    api = wandb.Api()
    
    # Try to find the artifact
    try:
        artifact = api.artifact(f"{run_path.rsplit('/', 1)[0]}/{artifact_name}:latest")
    except wandb.errors.CommError:
        # Try with full path
        artifact = api.artifact(artifact_name)
    
    artifact_dir = artifact.download(root=output_dir)
    print(f"Downloaded artifact to: {artifact_dir}")
    return artifact_dir


def load_model_output(artifact_dir: str) -> dict:
    """Load model output from downloaded artifact."""
    model_output_path = os.path.join(artifact_dir, 'model_output.pt')
    if os.path.exists(model_output_path):
        return torch.load(model_output_path, weights_only=False)
    
    # Check in subdirectories
    for root, dirs, files in os.walk(artifact_dir):
        if 'model_output.pt' in files:
            return torch.load(os.path.join(root, 'model_output.pt'), weights_only=False)
    
    raise FileNotFoundError(f"model_output.pt not found in {artifact_dir}")


def evaluate_from_artifact(
    artifact_dir: str,
    data_path: str,
    top_words: int = 10,
    output_path: str = None,
) -> dict:
    """Re-evaluate a model from downloaded artifact.
    
    Args:
        artifact_dir: Path to downloaded artifact
        data_path: Path to dataset or HuggingFace repo ID
        top_words: Number of top words per topic for evaluation
        output_path: Optional path to save evaluation results
        
    Returns:
        Evaluation results dictionary
    """
    # Load model output
    model_output = load_model_output(artifact_dir)
    print(f"Loaded model with {len(model_output['topics'])} topics")
    
    # Load training data for evaluation
    training_data = load_training_data(data_path, for_generative=False)
    
    # Get vocab embeddings
    vocab_embedding_path = os.path.join(training_data.local_path, 'vocab_embeddings.json')
    if os.path.exists(vocab_embedding_path):
        with open(vocab_embedding_path, encoding='utf-8') as f:
            vocab_embeddings = json.load(f)
    else:
        print("Computing vocab embeddings...")
        vocab_embeddings = get_openai_embedding(training_data.vocab)
        os.makedirs(os.path.dirname(vocab_embedding_path), exist_ok=True)
        with open(vocab_embedding_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_embeddings, f)
    
    # Evaluate
    print("Evaluating model...")
    results = evaluate_topic_model(
        model_output,
        top_words=top_words,
        test_corpus=training_data.bow_corpus,
        embeddings=vocab_embeddings,
        labels=training_data.labels,
    )
    
    # Print results
    print("\n=== Evaluation Results ===")
    for metric, value in results.items():
        print(f"  {metric}: {value}")
    
    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    return results


def list_artifacts(project: str, artifact_type: str = None) -> list:
    """List available artifacts in a wandb project.
    
    Args:
        project: Wandb project path (entity/project)
        artifact_type: Optional filter by artifact type
        
    Returns:
        List of artifact info dictionaries
    """
    api = wandb.Api()
    
    artifacts = []
    for artifact in api.artifacts(project, artifact_type or ""):
        artifacts.append({
            'name': artifact.name,
            'type': artifact.type,
            'version': artifact.version,
            'created_at': str(artifact.created_at),
        })
    
    return artifacts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download and evaluate wandb artifacts")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available artifacts')
    list_parser.add_argument('--project', type=str, required=True,
                             help='Wandb project path (entity/project)')
    list_parser.add_argument('--type', type=str, default=None,
                             help='Filter by artifact type')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download an artifact')
    download_parser.add_argument('--artifact', type=str, required=True,
                                 help='Artifact name or full path')
    download_parser.add_argument('--output_dir', type=str, default='./downloaded_artifacts',
                                 help='Directory to save artifact')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Download and evaluate an artifact')
    eval_parser.add_argument('--artifact', type=str, required=True,
                             help='Artifact name or full path (e.g., entity/project/artifact:version)')
    eval_parser.add_argument('--data_path', type=str, required=True,
                             help='Path to dataset or HuggingFace repo ID')
    eval_parser.add_argument('--top_words', type=int, default=10,
                             help='Number of top words per topic')
    eval_parser.add_argument('--output', type=str, default=None,
                             help='Path to save evaluation results')
    eval_parser.add_argument('--download_dir', type=str, default='./downloaded_artifacts',
                             help='Directory to download artifact')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        artifacts = list_artifacts(args.project, args.type)
        print(f"\n=== Artifacts in {args.project} ===")
        for a in artifacts:
            print(f"  {a['name']} (type: {a['type']}, version: {a['version']})")
    
    elif args.command == 'download':
        api = wandb.Api()
        artifact = api.artifact(args.artifact)
        artifact_dir = artifact.download(root=args.output_dir)
        print(f"Downloaded to: {artifact_dir}")
    
    elif args.command == 'evaluate':
        # Download artifact
        api = wandb.Api()
        artifact = api.artifact(args.artifact)
        artifact_dir = artifact.download(root=args.download_dir)
        
        # Evaluate
        evaluate_from_artifact(
            artifact_dir=artifact_dir,
            data_path=args.data_path,
            top_words=args.top_words,
            output_path=args.output,
        )
    
    else:
        parser.print_help()

