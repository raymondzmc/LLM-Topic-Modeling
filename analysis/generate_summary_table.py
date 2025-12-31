"""Generate summary table from wandb runs.

Fetches completed runs from wandb projects (one per dataset), filters by method,
and generates a summary table averaging metrics across K=25,50,75,100.
Includes statistical significance testing to highlight the best methods.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
import wandb
from collections import defaultdict
import numpy as np
from scipy import stats
from settings import settings

# Statistical significance threshold
SIGNIFICANCE_LEVEL = 0.05


# Configuration
DATASETS = ["20_newsgroups", "tweet_topic", "stackoverflow"]
K_VALUES = [25, 50, 75, 100]
REQUIRED_NUM_SEEDS = 5  # Only include runs with exactly this many seeds

# Baseline methods (config.model value)
BASELINE_METHODS = ["lda", "prodlda", "zeroshot", "combined", "etm", "bertopic", "ecrtm", "fastopic"]

# Generative LLM variants (extracted from run name: generative_{LLM_MODEL}_K{num})
GENERATIVE_LLM_MODELS = [
    "ERNIE-4.5-0.3B-PT",
    "Llama-3.1-8B-Instruct",
    "Llama-3.2-1B-Instruct",
]

# Metrics to extract (logged under avg/ prefix)
METRICS = ["cv_wiki", "llm_rating", "inverted_rbo", "purity"]
METRIC_KEYS = [f"avg/{m}" for m in METRICS]

# Retrieval metrics (logged under retrieval/ prefix)
RETRIEVAL_METRICS = ["precision@1", "precision@5", "precision@10"]
RETRIEVAL_METRIC_KEYS = [f"retrieval/avg_{m}" for m in RETRIEVAL_METRICS]

# Display names for methods
METHOD_DISPLAY_NAMES = {
    "lda": "LDA",
    "prodlda": "ProdLDA",
    "zeroshot": "ZeroShotTM",
    "combined": "CombinedTM",
    "etm": "ETM",
    "bertopic": "BERTopic",
    "ecrtm": "ECRTM",
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


def extract_seed_metrics_from_run(run, num_seeds: int = REQUIRED_NUM_SEEDS) -> dict:
    """Extract individual seed metrics from a run's summary.
    
    Returns:
        Dict: {metric_name: [seed_0_value, seed_1_value, ...]}
    """
    seed_metrics = {m: [] for m in METRICS}
    
    for seed in range(num_seeds):
        for metric in METRICS:
            key = f"seed_{seed}/{metric}"
            value = run.summary.get(key)
            if value is not None:
                seed_metrics[metric].append(value)
    
    return seed_metrics


def extract_retrieval_metrics_from_run(run) -> dict:
    """Extract the retrieval metrics from a run's summary.
    
    Returns:
        Dict: {metric_name: value} e.g., {'precision@1': 0.85, ...}
    """
    metrics = {}
    for metric in RETRIEVAL_METRICS:
        key = f"retrieval/avg_{metric}"
        value = run.summary.get(key)
        if value is not None:
            metrics[metric] = value
    return metrics


def extract_retrieval_seed_metrics_from_run(run, num_seeds: int = REQUIRED_NUM_SEEDS) -> dict:
    """Extract individual seed retrieval metrics from a run's summary.
    
    Returns:
        Dict: {metric_name: [seed_0_value, seed_1_value, ...]}
    """
    seed_metrics = {m: [] for m in RETRIEVAL_METRICS}
    
    for seed in range(num_seeds):
        for metric in RETRIEVAL_METRICS:
            key = f"retrieval/seed_{seed}/{metric}"
            value = run.summary.get(key)
            if value is not None:
                seed_metrics[metric].append(value)
    
    return seed_metrics


def aggregate_metrics(runs_data: list) -> dict:
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


def aggregate_seed_metrics(all_seed_metrics: list) -> dict:
    """Aggregate seed-level metrics across multiple runs.
    
    Args:
        all_seed_metrics: List of dicts, each containing {metric: [seed_values]}
        
    Returns:
        Dict: {metric: flat_list_of_all_values}
    """
    aggregated = {m: [] for m in METRICS}
    for seed_metrics in all_seed_metrics:
        for metric in METRICS:
            aggregated[metric].extend(seed_metrics.get(metric, []))
    return aggregated


def aggregate_retrieval_metrics(runs_data: list) -> dict:
    """Aggregate retrieval metrics across multiple runs by averaging.
    
    Args:
        runs_data: List of dicts with retrieval metric values
        
    Returns:
        Dict with averaged retrieval metric values
    """
    if not runs_data:
        return {m: None for m in RETRIEVAL_METRICS}
    
    aggregated = {}
    for metric in RETRIEVAL_METRICS:
        values = [r[metric] for r in runs_data if r.get(metric) is not None]
        if values:
            aggregated[metric] = np.mean(values)
        else:
            aggregated[metric] = None
    return aggregated


def aggregate_retrieval_seed_metrics(all_seed_metrics: list) -> dict:
    """Aggregate retrieval seed-level metrics across multiple runs.
    
    Args:
        all_seed_metrics: List of dicts, each containing {metric: [seed_values]}
        
    Returns:
        Dict: {metric: flat_list_of_all_values}
    """
    aggregated = {m: [] for m in RETRIEVAL_METRICS}
    for seed_metrics in all_seed_metrics:
        for metric in RETRIEVAL_METRICS:
            aggregated[metric].extend(seed_metrics.get(metric, []))
    return aggregated


def perform_significance_tests(raw_data: dict, method_keys: list, dataset: str, metric: str) -> dict:
    """Perform pairwise t-tests to find the best method and methods not significantly different.
    
    Args:
        raw_data: {method_key: {dataset: {metric: [values]}}}
        method_keys: List of method keys to compare
        dataset: Dataset name
        metric: Metric name
        
    Returns:
        Dict: {method_key: 'best' | 'not_sig_diff' | 'worse' | None}
    """
    # Collect values for each method
    method_values = {}
    for method_key in method_keys:
        values = raw_data.get(method_key, {}).get(dataset, {}).get(metric, [])
        if values:
            method_values[method_key] = np.array(values)
    
    if not method_values:
        return {m: None for m in method_keys}
    
    # Find the method with the highest mean
    means = {m: np.mean(v) for m, v in method_values.items()}
    best_method = max(means, key=means.get)
    best_values = method_values[best_method]
    
    results = {}
    for method_key in method_keys:
        if method_key not in method_values:
            results[method_key] = None
            continue
        
        if method_key == best_method:
            results[method_key] = 'best'
        else:
            # Perform two-sample t-test (independent samples)
            other_values = method_values[method_key]
            try:
                # Use Welch's t-test (unequal variances)
                _, p_value = stats.ttest_ind(best_values, other_values, equal_var=False)
                if p_value >= SIGNIFICANCE_LEVEL:
                    # Not significantly different from the best
                    results[method_key] = 'not_sig_diff'
                else:
                    results[method_key] = 'worse'
            except Exception:
                results[method_key] = 'worse'
    
    return results


def format_value_with_significance(value, sig_status, decimals=3):
    """Format a metric value with significance markers.
    
    Args:
        value: The metric value
        sig_status: 'best', 'not_sig_diff', 'worse', or None
        decimals: Number of decimal places
        
    Returns:
        Formatted string with markers:
        - **value** for best
        - *value* for not significantly different from best
        - value for worse
    """
    if value is None:
        return "-"
    
    formatted = f"{value:.{decimals}f}"
    
    if sig_status == 'best':
        return f"**{formatted}**"
    elif sig_status == 'not_sig_diff':
        return f"*{formatted}*"
    else:
        return formatted


def format_value(value, decimals=3):
    """Format a metric value for display."""
    if value is None:
        return "-"
    return f"{value:.{decimals}f}"


def print_summary_table(summary: dict, raw_data: dict = None):
    """Print the summary table in a readable format with significance markers.
    
    Args:
        summary: {method_key: {dataset: {metric: avg_value}}}
        raw_data: {method_key: {dataset: {metric: [all_values]}}} for significance testing
    """
    # Header
    metric_labels = ["CV", "LLM", "I-RBO", "Purity"]
    
    # Column widths (increased for significance markers)
    method_width = 25
    metric_width = 9
    dataset_width = metric_width * len(metric_labels) + len(metric_labels) - 1
    
    # Order of methods in the table
    method_order = BASELINE_METHODS + [f"generative_{llm}" for llm in GENERATIVE_LLM_MODELS]
    
    # Compute significance for each metric and dataset
    significance = {}
    if raw_data is not None:
        for dataset in DATASETS:
            significance[dataset] = {}
            for metric in METRICS:
                significance[dataset][metric] = perform_significance_tests(
                    raw_data, method_order, dataset, metric
                )
    
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
    
    # Print rows
    for method_key in method_order:
        display_name = METHOD_DISPLAY_NAMES.get(method_key, method_key)
        row = f"{display_name:<{method_width}}"
        
        for dataset in DATASETS:
            metrics_data = summary.get(method_key, {}).get(dataset, {})
            values = []
            for metric in METRICS:
                val = metrics_data.get(metric)
                sig_status = None
                if raw_data is not None:
                    sig_status = significance.get(dataset, {}).get(metric, {}).get(method_key)
                values.append(format_value_with_significance(val, sig_status))
            metrics_str = " ".join([f"{v:>{metric_width}}" for v in values])
            row += f" | {metrics_str}"
        
        print(row)
    
    print("=" * len(header1))
    print("Legend: **bold** = best, *italic* = not significantly different from best (p >= 0.05)")


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


def build_ablation_table(all_runs: dict):
    """Build the ablation summary table from wandb runs.
    
    Args:
        all_runs: Dict mapping dataset name to list of runs (pre-fetched)
    
    Returns:
        Tuple:
            - summary: {ablation_type: {dataset: {metric: avg_value}}}
            - raw_data: {ablation_type: {dataset: {metric: [all_values]}}}
    """
    results = defaultdict(lambda: defaultdict(list))
    raw_seed_data = defaultdict(lambda: defaultdict(list))
    
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
            
            # Check if run has the required number of seeds
            num_seeds = run.config.get("num_seeds", 0)
            if num_seeds != REQUIRED_NUM_SEEDS:
                continue
            
            # Extract average metrics
            metrics = extract_metrics_from_run(run)
            if metrics:
                results[ablation_key][dataset].append(metrics)
            
            # Extract seed-level metrics for statistical testing
            seed_metrics = extract_seed_metrics_from_run(run)
            if any(seed_metrics[m] for m in METRICS):
                raw_seed_data[ablation_key][dataset].append(seed_metrics)
    
    # Aggregate results
    summary = {}
    raw_data = {}
    for ablation_key in ABLATION_TYPES:
        summary[ablation_key] = {}
        raw_data[ablation_key] = {}
        for dataset in DATASETS:
            runs_data = results[ablation_key][dataset]
            summary[ablation_key][dataset] = aggregate_metrics(runs_data)
            
            # Aggregate seed-level data
            seed_data_list = raw_seed_data[ablation_key][dataset]
            raw_data[ablation_key][dataset] = aggregate_seed_metrics(seed_data_list)
            
            # Print debug info
            n_runs = len(runs_data)
            if n_runs > 0:
                print(f"  {ablation_key} on {dataset}: {n_runs} runs")
    
    return summary, raw_data


def print_ablation_table(summary: dict, raw_data: dict = None):
    """Print the ablation summary table in a readable format with significance markers.
    
    Args:
        summary: {ablation_type: {dataset: {metric: avg_value}}}
        raw_data: {ablation_type: {dataset: {metric: [all_values]}}} for significance testing
    """
    # Header
    metric_labels = ["CV", "LLM", "I-RBO", "Purity"]
    
    # Column widths (increased for significance markers)
    method_width = 25
    metric_width = 9
    dataset_width = metric_width * len(metric_labels) + len(metric_labels) - 1
    
    # Compute significance for each metric and dataset
    significance = {}
    if raw_data is not None:
        for dataset in DATASETS:
            significance[dataset] = {}
            for metric in METRICS:
                significance[dataset][metric] = perform_significance_tests(
                    raw_data, ABLATION_TYPES, dataset, metric
                )
    
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
                sig_status = None
                if raw_data is not None:
                    sig_status = significance.get(dataset, {}).get(metric, {}).get(ablation_key)
                values.append(format_value_with_significance(val, sig_status))
            metrics_str = " ".join([f"{v:>{metric_width}}" for v in values])
            row += f" | {metrics_str}"
        
        print(row)
    
    print("=" * len(header1))
    print("Legend: **bold** = best, *italic* = not significantly different from best (p >= 0.05)")


def build_summary_table_from_runs(all_runs: dict):
    """Build the complete summary table from pre-fetched wandb runs.
    
    Args:
        all_runs: Dict mapping dataset name to list of runs
    
    Returns:
        Tuple:
            - summary: {method_key: {dataset: {metric: avg_value}}}
            - raw_data: {method_key: {dataset: {metric: [all_values]}}}
    """
    results = defaultdict(lambda: defaultdict(list))
    raw_seed_data = defaultdict(lambda: defaultdict(list))
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
            
            # Check if run has the required number of seeds
            num_seeds = run.config.get("num_seeds", 0)
            if num_seeds != REQUIRED_NUM_SEEDS:
                continue
            
            # Extract average metrics
            metrics = extract_metrics_from_run(run)
            if metrics:
                results[method_key][dataset].append(metrics)
            
            # Extract seed-level metrics for statistical testing
            seed_metrics = extract_seed_metrics_from_run(run)
            if any(seed_metrics[m] for m in METRICS):
                raw_seed_data[method_key][dataset].append(seed_metrics)
    
    # Aggregate results
    summary = {}
    raw_data = {}
    for method_key in all_method_keys:
        summary[method_key] = {}
        raw_data[method_key] = {}
        for dataset in DATASETS:
            runs_data = results[method_key][dataset]
            summary[method_key][dataset] = aggregate_metrics(runs_data)
            
            # Aggregate seed-level data
            seed_data_list = raw_seed_data[method_key][dataset]
            raw_data[method_key][dataset] = aggregate_seed_metrics(seed_data_list)
            
            # Print debug info
            n_runs = len(runs_data)
            if n_runs > 0:
                print(f"  {method_key} on {dataset}: {n_runs} runs")
    
    return summary, raw_data


def build_retrieval_table_from_runs(all_runs: dict):
    """Build the retrieval summary table from pre-fetched wandb runs.
    
    Args:
        all_runs: Dict mapping dataset name to list of runs
    
    Returns:
        Tuple:
            - summary: {method_key: {dataset: {metric: avg_value}}}
            - raw_data: {method_key: {dataset: {metric: [all_values]}}}
    """
    results = defaultdict(lambda: defaultdict(list))
    raw_seed_data = defaultdict(lambda: defaultdict(list))
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
            
            # Check if run has the required number of seeds
            num_seeds = run.config.get("num_seeds", 0)
            if num_seeds != REQUIRED_NUM_SEEDS:
                continue
            
            # Extract retrieval metrics
            metrics = extract_retrieval_metrics_from_run(run)
            if metrics:
                results[method_key][dataset].append(metrics)
            
            # Extract seed-level retrieval metrics for statistical testing
            seed_metrics = extract_retrieval_seed_metrics_from_run(run)
            if any(seed_metrics[m] for m in RETRIEVAL_METRICS):
                raw_seed_data[method_key][dataset].append(seed_metrics)
    
    # Aggregate results
    summary = {}
    raw_data = {}
    for method_key in all_method_keys:
        summary[method_key] = {}
        raw_data[method_key] = {}
        for dataset in DATASETS:
            runs_data = results[method_key][dataset]
            summary[method_key][dataset] = aggregate_retrieval_metrics(runs_data)
            
            # Aggregate seed-level data
            seed_data_list = raw_seed_data[method_key][dataset]
            raw_data[method_key][dataset] = aggregate_retrieval_seed_metrics(seed_data_list)
            
            # Print debug info
            n_runs = len(runs_data)
            if n_runs > 0:
                print(f"  {method_key} on {dataset}: {n_runs} runs (retrieval)")
    
    return summary, raw_data


def perform_retrieval_significance_tests(raw_data: dict, method_keys: list, dataset: str, metric: str) -> dict:
    """Perform pairwise t-tests for retrieval metrics.
    
    Same as perform_significance_tests but for retrieval metrics.
    """
    # Collect values for each method
    method_values = {}
    for method_key in method_keys:
        values = raw_data.get(method_key, {}).get(dataset, {}).get(metric, [])
        if values:
            method_values[method_key] = np.array(values)
    
    if not method_values:
        return {m: None for m in method_keys}
    
    # Find the method with the highest mean
    means = {m: np.mean(v) for m, v in method_values.items()}
    best_method = max(means, key=means.get)
    best_values = method_values[best_method]
    
    results = {}
    for method_key in method_keys:
        if method_key not in method_values:
            results[method_key] = None
            continue
        
        if method_key == best_method:
            results[method_key] = 'best'
        else:
            # Perform two-sample t-test (independent samples)
            other_values = method_values[method_key]
            try:
                # Use Welch's t-test (unequal variances)
                _, p_value = stats.ttest_ind(best_values, other_values, equal_var=False)
                if p_value >= SIGNIFICANCE_LEVEL:
                    # Not significantly different from the best
                    results[method_key] = 'not_sig_diff'
                else:
                    results[method_key] = 'worse'
            except Exception:
                results[method_key] = 'worse'
    
    return results


def print_retrieval_table(summary: dict, raw_data: dict = None):
    """Print the retrieval summary table in a readable format with significance markers.
    
    Args:
        summary: {method_key: {dataset: {metric: avg_value}}}
        raw_data: {method_key: {dataset: {metric: [all_values]}}} for significance testing
    """
    # Header
    metric_labels = ["P@1", "P@5", "P@10"]
    
    # Column widths (increased for significance markers)
    method_width = 25
    metric_width = 9
    dataset_width = metric_width * len(metric_labels) + len(metric_labels) - 1
    
    # Order of methods in the table
    method_order = BASELINE_METHODS + [f"generative_{llm}" for llm in GENERATIVE_LLM_MODELS]
    
    # Compute significance for each metric and dataset
    significance = {}
    if raw_data is not None:
        for dataset in DATASETS:
            significance[dataset] = {}
            for metric in RETRIEVAL_METRICS:
                significance[dataset][metric] = perform_retrieval_significance_tests(
                    raw_data, method_order, dataset, metric
                )
    
    # Print header row 1 (dataset names)
    header1 = f"{'Method':<{method_width}}"
    for dataset in DATASETS:
        header1 += f" | {dataset:^{dataset_width}}"
    print("\n" + "=" * len(header1))
    print("RETRIEVAL EVALUATION (Precision@K)")
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
    for method_key in method_order:
        display_name = METHOD_DISPLAY_NAMES.get(method_key, method_key)
        row = f"{display_name:<{method_width}}"
        
        for dataset in DATASETS:
            metrics_data = summary.get(method_key, {}).get(dataset, {})
            values = []
            for metric in RETRIEVAL_METRICS:
                val = metrics_data.get(metric)
                sig_status = None
                if raw_data is not None:
                    sig_status = significance.get(dataset, {}).get(metric, {}).get(method_key)
                values.append(format_value_with_significance(val, sig_status))
            metrics_str = " ".join([f"{v:>{metric_width}}" for v in values])
            row += f" | {metrics_str}"
        
        print(row)
    
    print("=" * len(header1))
    print("Legend: **bold** = best, *italic* = not significantly different from best (p >= 0.05)")


def main():
    print("=" * 60)
    print("WandB Summary Table Generator")
    print("=" * 60)
    print(f"Entity: {settings.wandb_entity}")
    print(f"Datasets: {DATASETS}")
    print(f"K values: {K_VALUES}")
    print(f"Required seeds: {REQUIRED_NUM_SEEDS}")
    print(f"Metrics: {METRICS}")
    print(f"Significance level: {SIGNIFICANCE_LEVEL}")
    print("=" * 60 + "\n")
    
    # Fetch all runs once
    all_runs = {}
    for dataset in DATASETS:
        all_runs[dataset] = fetch_runs_for_dataset(dataset)
    
    # Build and print main summary table
    print("\n--- Main Methods Summary ---")
    summary, raw_data = build_summary_table_from_runs(all_runs)
    print_summary_table(summary, raw_data)
    
    # Build and print ablation table
    print("\n--- Ablation Experiments Summary ---")
    ablation_summary, ablation_raw_data = build_ablation_table(all_runs)
    print_ablation_table(ablation_summary, ablation_raw_data)
    
    # Build and print retrieval table
    print("\n--- Retrieval Evaluation Summary ---")
    retrieval_summary, retrieval_raw_data = build_retrieval_table_from_runs(all_runs)
    print_retrieval_table(retrieval_summary, retrieval_raw_data)


if __name__ == "__main__":
    main()

