"""Retrieval-based evaluation for topic models using topic distributions."""

import os
import json
import argparse
import tempfile
import torch
import numpy as np
import wandb
from tqdm import tqdm
from collections import defaultdict

from settings import settings


def compute_pairwise_kl_divergence_torch(P, device):
    """
    Compute pairwise KL divergence between rows of a matrix P using PyTorch.
    KL(P_i || P_j) = sum_k P_i[k] * (log P_i[k] - log P_j[k])
    This version computes the matrix row-by-row to be more memory-efficient.
    Args:
        P (torch.Tensor): Input tensor of shape (n_docs, n_features) where rows are distributions.
        device (torch.device): The device (CPU or CUDA) to perform calculations on.
    Returns:
        torch.Tensor: Pairwise KL divergence matrix of shape (n_docs, n_docs).
    """
    P = P.to(device)
    n_docs, n_features = P.shape
    P_norm = P / P.sum(dim=1, keepdims=True)
    P_stable = P_norm + 1e-12
    logP_all = P_stable.log()
    kl_matrix = torch.zeros((n_docs, n_docs))

    print(f"  Calculating KL divergence row-by-row for {n_docs} documents...")
    for i in tqdm(range(n_docs), desc="  KL Div Row", unit="doc", leave=False, dynamic_ncols=True):
        log_diff = logP_all[i, :].unsqueeze(0) - logP_all
        product = P_norm[i, :].unsqueeze(0) * log_diff
        kl_matrix[i, :] = product.sum(dim=-1).cpu()
    kl_matrix.fill_diagonal_(0)
    return kl_matrix


def compute_precision_at_k(retrieved_indices, query_label, all_labels, k_values=[1, 5, 10]):
    """Compute precision@k for the retrieved documents."""
    results = {}
    
    for k in k_values:
        if k > len(retrieved_indices):
            continue
        top_k_indices = retrieved_indices[:k]
        top_k_labels = all_labels[top_k_indices]
        precision = np.mean(top_k_labels == query_label)
        results[f'precision@{k}'] = precision
        
    return results


def apply_subsetting(labels, retrieval_representation, subset_size):
    """Apply subsetting for an even number of documents per label."""
    label_to_indices = defaultdict(list)
    for i, label in enumerate(labels):
        label_to_indices[label].append(i)
        
    min_count = min(len(indices) for indices in label_to_indices.values())
    if subset_size < len(label_to_indices) * min_count:
        docs_per_label = subset_size // len(label_to_indices)
        min_count = min(min_count, docs_per_label)
        
    subset_indices = sum([indices[:min_count] for indices in label_to_indices.values()], [])
    labels = labels[subset_indices]
    retrieval_representation = retrieval_representation[subset_indices]
    return labels, retrieval_representation


def fetch_finished_runs(wandb_project: str) -> list:
    """
    Fetch all finished runs from a wandb project.
    
    Args:
        wandb_project: Wandb project name
        
    Returns:
        List of run objects with state='finished'
    """
    api = wandb.Api()
    
    print(f"Fetching finished runs from {settings.wandb_entity}/{wandb_project}...")
    runs = api.runs(
        f"{settings.wandb_entity}/{wandb_project}",
        filters={"state": "finished"},
        order="-created_at",
    )
    runs_list = list(runs)
    print(f"Found {len(runs_list)} finished runs")
    
    return runs_list


def has_retrieval_artifact(source_run) -> bool:
    """
    Check if a run already has a retrieval evaluation artifact.
    
    Args:
        source_run: Wandb run object
        
    Returns:
        True if retrieval artifact exists, False otherwise
    """
    artifacts = list(source_run.logged_artifacts())
    retrieval_artifacts = [a for a in artifacts if a.type == "evaluation" and "retrieval" in a.name]
    return len(retrieval_artifacts) > 0


def download_artifact_for_run(source_run, temp_dir: str):
    """
    Download model artifact for a given run object.
    
    Args:
        source_run: Wandb run object
        temp_dir: Temporary directory to download artifact to
        
    Returns:
        Tuple of (artifact_dir, metadata) or (None, None) if no artifact found
    """
    # Find model artifact
    artifacts = list(source_run.logged_artifacts())
    model_artifacts = [a for a in artifacts if a.type == "model"]
    
    if len(model_artifacts) == 0:
        return None, None
    
    artifact = model_artifacts[-1]
    print(f"  Found artifact: {artifact.name} (v{artifact.version})")
    
    # Download artifact
    artifact_dir = artifact.download(root=temp_dir)
    
    # Get metadata
    metadata = artifact.metadata or {}
    
    return artifact_dir, metadata


def download_wandb_artifact(run_id_or_name: str, wandb_project: str, temp_dir: str):
    """
    Download model artifact from a wandb run.
    
    Args:
        run_id_or_name: Wandb run ID or display name
        wandb_project: Wandb project name
        temp_dir: Temporary directory to download artifact to
        
    Returns:
        Tuple of (artifact_dir, metadata, source_run)
    """
    api = wandb.Api()
    
    # Find run by ID first, then by name
    source_run = None
    try:
        source_run = api.run(f"{settings.wandb_entity}/{wandb_project}/{run_id_or_name}")
        print(f"Found run by ID: {source_run.name} ({source_run.id})")
    except wandb.errors.CommError:
        print(f"Run ID '{run_id_or_name}' not found, searching by name...")
        runs = api.runs(
            f"{settings.wandb_entity}/{wandb_project}",
            filters={"display_name": run_id_or_name},
            order="-created_at",
        )
        runs_list = list(runs)
        
        if len(runs_list) == 0:
            raise ValueError(f"No run found with ID or name: {run_id_or_name}")
        
        if len(runs_list) > 1:
            print(f"WARNING: Found {len(runs_list)} runs with name '{run_id_or_name}', using most recent")
            for i, r in enumerate(runs_list[:5]):
                print(f"   {i+1}. ID: {r.id}, Created: {r.created_at}")
        
        source_run = runs_list[0]
        print(f"Using run: {source_run.name} ({source_run.id})")
    
    # Find model artifact
    print("\nSearching for model artifact...")
    artifacts = list(source_run.logged_artifacts())
    model_artifacts = [a for a in artifacts if a.type == "model"]
    
    if len(model_artifacts) == 0:
        raise ValueError(f"No model artifacts found for run {source_run.id}")
    
    artifact = model_artifacts[-1]
    print(f"Found artifact: {artifact.name} (v{artifact.version})")
    
    # Download artifact
    artifact_dir = artifact.download(root=temp_dir)
    print(f"Downloaded to: {artifact_dir}")
    
    # Get metadata
    metadata = artifact.metadata or {}
    
    return artifact_dir, metadata, source_run


def evaluate_single_seed(
    model_output_path: str,
    labels: np.ndarray,
    subset_size: int = -1,
    device: torch.device = None,
) -> dict:
    """
    Evaluate retrieval metrics for a single seed's model output.
    
    Args:
        model_output_path: Path to model_output.pt file
        labels: Document labels (already filtered for empty docs)
        subset_size: Number of documents to subset (-1 for all)
        device: Torch device to use
        
    Returns:
        Dictionary with precision@k results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model output
    output = torch.load(model_output_path, weights_only=False)
    if 'topic-document-matrix' not in output.keys():
        raise ValueError(f"topic-document-matrix not found in output keys at {model_output_path}")
    
    topic_distribution_np: np.ndarray = output['topic-document-matrix'].transpose()
    
    # Validate alignment with labels
    if topic_distribution_np.shape[0] != len(labels):
        raise ValueError(
            f"Shape mismatch between topic distributions ({topic_distribution_np.shape[0]}) "
            f"and labels ({len(labels)}). The model was likely trained with filtered data."
        )
    
    # Apply subsetting if requested
    eval_labels = labels.copy()
    if subset_size > 0:
        eval_labels, topic_distribution_np = apply_subsetting(
            eval_labels, topic_distribution_np, subset_size
        )
    
    print(f"  Evaluating {topic_distribution_np.shape[0]} documents...")
    
    # Compute pairwise KL divergence
    topic_distribution_tensor = torch.from_numpy(topic_distribution_np).float()
    kl_matrix = compute_pairwise_kl_divergence_torch(topic_distribution_tensor, device)
    similarity_matrix = kl_matrix.cpu().numpy()
    
    # Calculate precision@k for each document
    n_docs = similarity_matrix.shape[0]
    k_values = [1, 5, 10]
    max_k = max(k_values)
    
    precision_results = {k: [] for k in k_values}
    
    for i in tqdm(range(n_docs), desc="  Precision@k", leave=False, dynamic_ncols=True):
        # Get indices sorted by KL divergence (ascending, smaller is more similar)
        retrieved_indices = np.argsort(similarity_matrix[i])
        # Remove self
        retrieved_indices = retrieved_indices[retrieved_indices != i][:max_k]
        
        precisions = compute_precision_at_k(retrieved_indices, eval_labels[i], eval_labels, k_values)
        
        for k_val, precision in precisions.items():
            precision_results[int(k_val.split('@')[1])].append(precision)
    
    # Compute mean precision for each k
    results = {}
    for k in k_values:
        if precision_results[k]:
            results[f'precision@{k}'] = float(np.mean(precision_results[k]))
    
    return results


def evaluate_run(source_run, artifact_dir: str, metadata: dict, args, device: torch.device) -> bool:
    """
    Evaluate a single run's retrieval metrics.
    
    Args:
        source_run: Wandb run object
        artifact_dir: Path to downloaded artifact
        metadata: Artifact metadata
        args: Command line arguments
        device: Torch device to use
        
    Returns:
        True if evaluation succeeded, False otherwise
    """
    # Get metadata
    num_seeds = metadata.get('num_seeds', 1)
    model_name = metadata.get('model', 'unknown')
    dataset_name = metadata.get('dataset', 'unknown')
    num_topics = metadata.get('num_topics', 0)
    
    print(f"\nMetadata: model={model_name}, dataset={dataset_name}, K={num_topics}, seeds={num_seeds}")
    
    # Load labels from artifact
    labels_path = os.path.join(artifact_dir, 'labels.json')
    if not os.path.exists(labels_path):
        raise ValueError(f"labels.json not found in artifact at {artifact_dir}")
    
    print("Loading labels from artifact...")
    with open(labels_path, encoding='utf-8') as f:
        labels = np.array(json.load(f))
    print(f"  Loaded {len(labels)} labels")
    
    # Evaluate all seeds
    all_seed_results = defaultdict(list)
    results_dir = os.path.join(artifact_dir, 'retrieval_results')
    os.makedirs(results_dir, exist_ok=True)
    
    for seed in range(num_seeds):
        seed_dir = os.path.join(artifact_dir, f"seed_{seed}")
        model_output_path = os.path.join(seed_dir, 'model_output.pt')
        
        if not os.path.exists(model_output_path):
            print(f"[Seed {seed}] model_output.pt not found, skipping")
            continue
        
        print(f"\n[Seed {seed}] Evaluating...")
        seed_results = evaluate_single_seed(
            model_output_path=model_output_path,
            labels=labels,
            subset_size=args.subset_size,
            device=device,
        )
        
        # Store results
        for k, precision in seed_results.items():
            all_seed_results[k].append(precision)
            print(f"  {k}: {precision:.4f}")
        
        # Save per-seed results
        seed_result_path = os.path.join(results_dir, f'seed_{seed}_results.json')
        with open(seed_result_path, 'w', encoding='utf-8') as f:
            json.dump(seed_results, f, indent=2)
    
    # Aggregate results across seeds
    if not all_seed_results:
        print("\nNo results to aggregate!")
        return False
    
    aggregated_results = {}
    print(f"\n{'='*60}")
    print(f"AVERAGE RESULTS ACROSS {len(all_seed_results[list(all_seed_results.keys())[0]])} SEEDS")
    print(f"Model: {model_name}, Dataset: {dataset_name}, K: {num_topics}")
    print(f"{'='*60}")
    
    for k in sorted(all_seed_results.keys()):
        values = all_seed_results[k]
        avg = float(np.mean(values))
        std = float(np.std(values))
        aggregated_results[f'avg_{k}'] = avg
        aggregated_results[f'std_{k}'] = std
        print(f"  {k}: {avg:.4f} ± {std:.4f}")
    
    # Save aggregated results
    aggregated_path = os.path.join(results_dir, 'aggregated_results.json')
    with open(aggregated_path, 'w', encoding='utf-8') as f:
        json.dump(aggregated_results, f, indent=2)
    
    # Upload results to wandb by resuming the original run
    print("\nUploading results to wandb...")
    print(f"Resuming run: {source_run.name} ({source_run.id})")
    
    wb_run = wandb.init(
        project=args.wandb_project,
        entity=settings.wandb_entity,
        id=source_run.id,
        resume="allow",
        mode='online' if not args.wandb_offline else 'offline',
    )
    
    # Log metrics
    for k, v in aggregated_results.items():
        wb_run.log({f"retrieval/{k}": v})
    
    # Log per-seed results
    for k in sorted(all_seed_results.keys()):
        for seed_idx, precision in enumerate(all_seed_results[k]):
            wb_run.log({f"retrieval/seed_{seed_idx}/{k}": precision})
    
    # Create and upload artifact
    artifact = wandb.Artifact(
        name=f"{model_name}-K{num_topics}-{dataset_name}-retrieval",
        type="evaluation",
        description=f"Retrieval evaluation results",
        metadata={
            "model": model_name,
            "dataset": dataset_name,
            "num_topics": num_topics,
            "num_seeds": num_seeds,
            "subset_size": args.subset_size,
        }
    )
    artifact.add_dir(results_dir)
    wb_run.log_artifact(artifact)
    
    wb_run.finish()
    
    print(f"\nResults uploaded to run: https://wandb.ai/{settings.wandb_entity}/{args.wandb_project}/runs/{source_run.id}")
    
    return True


def main(args):
    """Main function to evaluate retrieval metrics from wandb artifacts."""
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.all:
        # Batch mode: evaluate all finished runs in the project
        print(f"\n{'='*60}")
        print(f"Batch Retrieval Evaluation")
        print(f"Project: {settings.wandb_entity}/{args.wandb_project}")
        print(f"Mode: All finished runs")
        print(f"{'='*60}\n")
        
        runs = fetch_finished_runs(args.wandb_project)
        
        if not runs:
            print("No finished runs found!")
            return
        
        success_count = 0
        error_count = 0
        skipped_count = 0
        
        already_evaluated_count = 0
        
        for i, run in enumerate(runs):
            print(f"\n{'#'*60}")
            print(f"[{i+1}/{len(runs)}] Evaluating run: {run.name} ({run.id})")
            print(f"{'#'*60}")
            
            # Check if already evaluated (unless force_recompute is set)
            if not args.force_recompute and has_retrieval_artifact(run):
                print(f"  Retrieval artifact already exists, skipping (use --force_recompute to override)")
                already_evaluated_count += 1
                continue
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download artifact
                artifact_dir, metadata = download_artifact_for_run(run, temp_dir)
                
                if artifact_dir is None:
                    print(f"  No model artifact found, skipping")
                    skipped_count += 1
                    continue
                
                # Evaluate
                try:
                    success = evaluate_run(run, artifact_dir, metadata, args, device)
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    print(f"  ERROR: {e}")
                    raise  # Re-raise the error as requested
        
        print(f"\n{'='*60}")
        print(f"Batch Evaluation Complete")
        print(f"  Successful: {success_count}")
        print(f"  Errors: {error_count}")
        print(f"  Skipped (no model artifact): {skipped_count}")
        print(f"  Skipped (already evaluated): {already_evaluated_count}")
        print(f"{'='*60}")
    
    else:
        # Single run mode
        if args.run_id_or_name is None:
            raise ValueError("--run_id_or_name is required when --all is not set")
        
        print(f"\n{'='*60}")
        print(f"Retrieval Evaluation")
        print(f"Project: {settings.wandb_entity}/{args.wandb_project}")
        print(f"Run: {args.run_id_or_name}")
        print(f"{'='*60}\n")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download wandb artifact
            artifact_dir, metadata, source_run = download_wandb_artifact(
                args.run_id_or_name,
                args.wandb_project,
                temp_dir,
            )
            
            # Check if already evaluated (unless force_recompute is set)
            if not args.force_recompute and has_retrieval_artifact(source_run):
                print(f"Retrieval artifact already exists, skipping (use --force_recompute to override)")
                return
            
            evaluate_run(source_run, artifact_dir, metadata, args, device)
        
        print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate topic model retrieval from wandb artifacts")
    
    # Required arguments
    parser.add_argument("--wandb_project", type=str, required=True,
                        help="Wandb project name")
    parser.add_argument("--run_id_or_name", type=str, default=None,
                        help="Wandb run ID or display name to evaluate (required if --all not set)")
    
    # Batch mode
    parser.add_argument("--all", action='store_true',
                        help="Evaluate all finished runs in the project")
    parser.add_argument("--force_recompute", action='store_true',
                        help="Re-compute metrics even if retrieval artifact already exists")
    
    # Optional arguments
    parser.add_argument("--subset_size", type=int, default=-1,
                        help="Number of documents to use for subset. Default is -1 (use all)")
    parser.add_argument("--wandb_offline", action='store_true',
                        help="Run wandb in offline mode")
    
    args = parser.parse_args()
    main(args)
