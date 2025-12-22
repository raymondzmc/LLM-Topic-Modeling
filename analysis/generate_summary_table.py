"""Generate summary table from wandb runs.

Fetches completed runs from wandb projects (one per dataset), filters by method,
and generates a summary table averaging metrics across K=25,50,75,100.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
import wandb
from collections import defaultdict
import numpy as np
from settings import settings


# Configuration
DATASETS = ["20_newsgroups", "tweet_topic", "stackoverflow"]
K_VALUES = [25, 50, 75, 100]

# Baseline methods (config.model value)
BASELINE_METHODS = ["lda", "prodlda", "zeroshot", "combined", "etm", "bertopic", "fastopic"]

# Generative LLM variants (extracted from run name: generative_{LLM_MODEL}_K{num})
GENERATIVE_LLM_MODELS = [
    "ERNIE-4.5-0.3B-PT",
    "Llama-3.1-8B-Instruct",
    "Llama-3.2-1B-Instruct",
]

# Metrics to extract (logged under avg/ prefix)
METRICS = ["cv", "llm_rating", "inverted_rbo", "purity"]
METRIC_KEYS = [f"avg/{m}" for m in METRICS]

# Display names for methods
METHOD_DISPLAY_NAMES = {
    "lda": "LDA",
    "prodlda": "ProdLDA",
    "zeroshot": "ZeroShotTM",
    "combined": "CombinedTM",
    "etm": "ETM",
    "bertopic": "BERTopic",
    "fastopic": "FASTopic",
    "generative_ERNIE-4.5-0.3B-PT": "Generative (ERNIE)",
    "generative_Llama-3.1-8B-Instruct": "Generative (Llama-8B)",
    "generative_Llama-3.2-1B-Instruct": "Generative (Llama-1B)",
}

# Ablation experiment configuration (ERNIE only)
ABLATION_LLM_MODEL = "ERNIE-4.5-0.3B-PT"
ABLATION_TYPES = ["original", "bow_target", "contextualized_embeddings", "nll_loss"]
ABLATION_DISPLAY_NAMES = {
    "original": "Original (Ours)",
    "bow_target": "BoW Target",
    "contextualized_embeddings": "Contextualized Embeddings",
    "nll_loss": "NLL Loss",
}


def get_method_key_from_run(run):
    """Extract method key from a wandb run.
    
    For baseline methods, returns config.model.
    For generative methods, returns 'generative_{LLM_MODEL}'.
    """
    model = run.config.get("model", "")
    
    if model == "generative":
        # Parse run name to extract LLM model: generative_{LLM_MODEL}_K{num}
        run_name = run.name
        for llm_model in GENERATIVE_LLM_MODELS:
            if f"generative_{llm_model}_K" in run_name:
                return f"generative_{llm_model}"
        # If no match found, return None to skip this run
        return None
    elif model in BASELINE_METHODS:
        return model
    else:
        return None


def fetch_runs_for_dataset(dataset: str) -> list:
    """Fetch all finished runs for a dataset from wandb."""
    api = wandb.Api()
    project_path = f"{settings.wandb_entity}/{dataset}"
    
    print(f"Fetching runs from {project_path}...")
    
    try:
        runs = api.runs(
            project_path,
            filters={"state": "finished"},
            order="-created_at",
        )
        runs_list = list(runs)
        print(f"  Found {len(runs_list)} finished runs")
        return runs_list
    except Exception as e:
        print(f"  Error fetching runs: {e}")
        return []


def extract_metrics_from_run(run) -> dict:
    """Extract the required metrics from a run's summary."""
    metrics = {}
    for metric_key in METRIC_KEYS:
        value = run.summary.get(metric_key)
        if value is not None:
            # Extract short name (e.g., 'cv' from 'avg/cv')
            short_name = metric_key.split("/")[1]
            metrics[short_name] = value
    return metrics


def aggregate_metrics(runs_data: list[dict]) -> dict:
    """Aggregate metrics across multiple runs by averaging.
    
    Args:
        runs_data: List of dicts with metric values
        
    Returns:
        Dict with averaged metric values
    """
    if not runs_data:
        return {m: None for m in METRICS}
    
    aggregated = {}
    for metric in METRICS:
        values = [r[metric] for r in runs_data if r.get(metric) is not None]
        if values:
            aggregated[metric] = np.mean(values)
        else:
            aggregated[metric] = None
    return aggregated


def format_value(value, decimals=3):
    """Format a metric value for display."""
    if value is None:
        return "-"
    return f"{value:.{decimals}f}"


def print_summary_table(summary: dict):
    """Print the summary table in a readable format."""
    # Header
    metric_labels = ["CV", "LLM", "I-RBO", "Purity"]
    
    # Column widths
    method_width = 25
    metric_width = 7
    dataset_width = metric_width * len(metric_labels) + len(metric_labels) - 1
    
    # Print header row 1 (dataset names)
    header1 = f"{'Method':<{method_width}}"
    for dataset in DATASETS:
        header1 += f" | {dataset:^{dataset_width}}"
    print("\n" + "=" * len(header1))
    print(header1)
    
    # Print header row 2 (metric names)
    header2 = " " * method_width
    for _ in DATASETS:
        metrics_str = " ".join([f"{m:>{metric_width}}" for m in metric_labels])
        header2 += f" | {metrics_str}"
    print(header2)
    print("=" * len(header1))
    
    # Order of methods in the table
    method_order = BASELINE_METHODS + [f"generative_{llm}" for llm in GENERATIVE_LLM_MODELS]
    
    # Print rows
    for method_key in method_order:
        display_name = METHOD_DISPLAY_NAMES.get(method_key, method_key)
        row = f"{display_name:<{method_width}}"
        
        for dataset in DATASETS:
            metrics_data = summary.get(method_key, {}).get(dataset, {})
            values = []
            for metric in METRICS:
                val = metrics_data.get(metric)
                values.append(format_value(val))
            metrics_str = " ".join([f"{v:>{metric_width}}" for v in values])
            row += f" | {metrics_str}"
        
        print(row)
    
    print("=" * len(header1))


def get_ablation_key_from_run(run) -> Optional[str]:
    """Extract ablation type key from a wandb run.
    
    Only considers ERNIE generative runs.
    Returns None for non-ERNIE runs.
    
    Run name patterns:
    - Original: generative_ERNIE-4.5-0.3B-PT_K{num} (no ablation suffix)
    - BoW Target: generative_ERNIE-4.5-0.3B-PT_K{num}_bow-target_CE
    - Contextualized Embeddings: generative_ERNIE-4.5-0.3B-PT_K{num}_gte-large-en-v1.5
    - NLL Loss: generative_ERNIE-4.5-0.3B-PT_K{num}_CE (only _CE, no other suffix)
    """
    model = run.config.get("model", "")
    if model != "generative":
        return None
    
    run_name = run.name
    
    # Only consider ERNIE runs
    if ABLATION_LLM_MODEL not in run_name:
        return None
    
    # Check for ablation patterns (order matters - check more specific patterns first)
    if "_bow-target" in run_name:
        return "bow_target"
    elif "_gte-large-en-v1.5" in run_name:
        return "contextualized_embeddings"
    elif run_name.endswith("_CE"):
        # Only _CE suffix, no bow-target or embedding suffix
        return "nll_loss"
    else:
        # No ablation suffix - this is the original model
        return "original"


def build_ablation_table(all_runs: dict) -> dict:
    """Build the ablation summary table from wandb runs.
    
    Args:
        all_runs: Dict mapping dataset name to list of runs (pre-fetched)
    
    Returns:
        Dict: {ablation_type: {dataset: {metric: value}}}
    """
    results = defaultdict(lambda: defaultdict(list))
    
    for dataset in DATASETS:
        runs = all_runs.get(dataset, [])
        
        for run in runs:
            # Get ablation key
            ablation_key = get_ablation_key_from_run(run)
            if ablation_key is None:
                continue
            
            # Check if K is in our target values
            num_topics = run.config.get("num_topics")
            if num_topics not in K_VALUES:
                continue
            
            # Extract metrics
            metrics = extract_metrics_from_run(run)
            if metrics:
                results[ablation_key][dataset].append(metrics)
    
    # Aggregate results
    summary = {}
    for ablation_key in ABLATION_TYPES:
        summary[ablation_key] = {}
        for dataset in DATASETS:
            runs_data = results[ablation_key][dataset]
            summary[ablation_key][dataset] = aggregate_metrics(runs_data)
            
            # Print debug info
            n_runs = len(runs_data)
            if n_runs > 0:
                print(f"  {ablation_key} on {dataset}: {n_runs} runs")
    
    return summary


def print_ablation_table(summary: dict):
    """Print the ablation summary table in a readable format."""
    # Header
    metric_labels = ["CV", "LLM", "I-RBO", "Purity"]
    
    # Column widths
    method_width = 25
    metric_width = 7
    dataset_width = metric_width * len(metric_labels) + len(metric_labels) - 1
    
    # Print header row 1 (dataset names)
    header1 = f"{'Ablation':<{method_width}}"
    for dataset in DATASETS:
        header1 += f" | {dataset:^{dataset_width}}"
    print("\n" + "=" * len(header1))
    print("ABLATION EXPERIMENTS (ERNIE only)")
    print("=" * len(header1))
    print(header1)
    
    # Print header row 2 (metric names)
    header2 = " " * method_width
    for _ in DATASETS:
        metrics_str = " ".join([f"{m:>{metric_width}}" for m in metric_labels])
        header2 += f" | {metrics_str}"
    print(header2)
    print("=" * len(header1))
    
    # Print rows
    for ablation_key in ABLATION_TYPES:
        display_name = ABLATION_DISPLAY_NAMES.get(ablation_key, ablation_key)
        row = f"{display_name:<{method_width}}"
        
        for dataset in DATASETS:
            metrics_data = summary.get(ablation_key, {}).get(dataset, {})
            values = []
            for metric in METRICS:
                val = metrics_data.get(metric)
                values.append(format_value(val))
            metrics_str = " ".join([f"{v:>{metric_width}}" for v in values])
            row += f" | {metrics_str}"
        
        print(row)
    
    print("=" * len(header1))


def build_summary_table_from_runs(all_runs: dict) -> dict:
    """Build the complete summary table from pre-fetched wandb runs.
    
    Args:
        all_runs: Dict mapping dataset name to list of runs
    
    Returns:
        Dict: {method_key: {dataset: {metric: value}}}
    """
    results = defaultdict(lambda: defaultdict(list))
    all_method_keys = BASELINE_METHODS + [f"generative_{llm}" for llm in GENERATIVE_LLM_MODELS]
    
    for dataset in DATASETS:
        runs = all_runs.get(dataset, [])
        
        for run in runs:
            # Get method key
            method_key = get_method_key_from_run(run)
            if method_key is None:
                continue
            
            # Check if K is in our target values
            num_topics = run.config.get("num_topics")
            if num_topics not in K_VALUES:
                continue
            
            # Extract metrics
            metrics = extract_metrics_from_run(run)
            if metrics:
                results[method_key][dataset].append(metrics)
    
    # Aggregate results
    summary = {}
    for method_key in all_method_keys:
        summary[method_key] = {}
        for dataset in DATASETS:
            runs_data = results[method_key][dataset]
            summary[method_key][dataset] = aggregate_metrics(runs_data)
            
            # Print debug info
            n_runs = len(runs_data)
            if n_runs > 0:
                print(f"  {method_key} on {dataset}: {n_runs} runs")
    
    return summary


def main():
    print("=" * 60)
    print("WandB Summary Table Generator")
    print("=" * 60)
    print(f"Entity: {settings.wandb_entity}")
    print(f"Datasets: {DATASETS}")
    print(f"K values: {K_VALUES}")
    print(f"Metrics: {METRICS}")
    print("=" * 60 + "\n")
    
    # Fetch all runs once
    all_runs = {}
    for dataset in DATASETS:
        all_runs[dataset] = fetch_runs_for_dataset(dataset)
    
    # Build and print main summary table
    print("\n--- Main Methods Summary ---")
    summary = build_summary_table_from_runs(all_runs)
    print_summary_table(summary)
    
    # Build and print ablation table
    print("\n--- Ablation Experiments Summary ---")
    ablation_summary = build_ablation_table(all_runs)
    print_ablation_table(ablation_summary)


if __name__ == "__main__":
    main()

