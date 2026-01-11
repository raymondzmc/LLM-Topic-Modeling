"""Summarize vocab size experiments from wandb.

Fetches completed runs from vocab size experiment projects, validates completeness,
and exports results to CSV for academic paper analysis.

Projects:
- 20_newsgroups_vocab_500
- 20_newsgroups_vocab_1000
- 20_newsgroups_vocab_2000 (or 20_newsgroups for default)
- 20_newsgroups_vocab_4000
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import wandb
from collections import defaultdict
from settings import settings

# Configuration
VOCAB_SIZES = [500, 1000, 2000, 4000]
K_VALUES = [25, 50, 75, 100]
REQUIRED_NUM_SEEDS = 5

# Project name mapping
# vocab_size 2000 uses the main project name, others use vocab-specific projects
def get_project_name(vocab_size: int) -> str:
    if vocab_size == 2000:
        return "20_newsgroups"
    return f"20_newsgroups_vocab_{vocab_size}"

# Baseline methods
BASELINE_METHODS = ["lda", "prodlda", "zeroshot", "combined", "etm", "bertopic", "ecrtm", "fastopic"]

# Generative LLM model for this experiment
LLM_MODEL = "ERNIE-4.5-0.3B-PT"

# All metrics to extract
METRICS = ["cv_wiki", "llm_rating", "inverted_rbo", "purity"]
METRIC_KEYS = [f"avg/{m}" for m in METRICS]

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
    f"generative_{LLM_MODEL}": f"Generative ({LLM_MODEL})",
}


def get_method_key_from_run(run) -> str | None:
    """Extract method key from a wandb run.
    
    For baseline methods, returns config.model only if run name exactly matches {model}_K{num}.
    For generative methods, returns 'generative_{LLM_MODEL}' only if run name exactly matches.
    This ensures we exclude ablation variants and other experiment runs.
    """
    model = run.config.get("model", "")
    run_name = run.name
    
    if model == "generative":
        # Parse run name to extract LLM model: generative_{LLM_MODEL}_K{num}
        expected_prefix = f"generative_{LLM_MODEL}_K"
        if run_name.startswith(expected_prefix):
            # Extract the part after the prefix (should be just a number for original runs)
            suffix = run_name[len(expected_prefix):]
            # Original runs have only the K number, no additional suffix
            if suffix.isdigit():
                return f"generative_{LLM_MODEL}"
        # Otherwise it's an ablation variant or different LLM - skip
        return None
    elif model in BASELINE_METHODS:
        # For baselines, require exact name match: {model}_K{num}
        # This excludes ablation variants and other experiments
        expected_prefix = f"{model}_K"
        if run_name.startswith(expected_prefix):
            suffix = run_name[len(expected_prefix):]
            # Original runs have only the K number, no additional suffix
            if suffix.isdigit():
                return model
        # Otherwise it's not a standard baseline run - skip
        return None
    else:
        return None


def fetch_runs_for_project(project_name: str) -> list:
    """Fetch all finished runs for a project from wandb."""
    api = wandb.Api()
    project_path = f"{settings.wandb_entity}/{project_name}"
    
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
    """Extract all metrics from a run's summary."""
    metrics = {}
    
    # Extract main metrics
    for metric in METRICS:
        key = f"avg/{metric}"
        value = run.summary.get(key)
        if value is not None:
            metrics[metric] = value
    
    # Extract training time
    training_time = run.summary.get("avg/training_time")
    if training_time is not None:
        metrics["training_time"] = training_time
    
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


def process_vocab_size_runs(vocab_size: int, debug: bool = False) -> tuple[list[dict], dict]:
    """Process runs for a specific vocab size.
    
    Returns:
        Tuple of (results_list, missing_dict)
        - results_list: List of dicts with run data
        - missing_dict: Dict mapping method_key to list of missing K values
    """
    project_name = get_project_name(vocab_size)
    runs = fetch_runs_for_project(project_name)
    
    # Track runs by (method_key, K) - take only the latest for each combination
    seen_combinations = set()
    results = []
    method_k_found = defaultdict(set)  # method_key -> set of K values found
    
    # All expected methods
    all_methods = BASELINE_METHODS + [f"generative_{LLM_MODEL}"]
    
    # Debug: track skip reasons
    skip_reasons = defaultdict(int)
    
    for run in runs:
        method_key = get_method_key_from_run(run)
        if method_key is None:
            skip_reasons["name_mismatch"] += 1
            if debug:
                print(f"    SKIP (name): {run.name}")
            continue
        
        num_topics = run.config.get("num_topics")
        if num_topics not in K_VALUES:
            skip_reasons["k_not_in_range"] += 1
            if debug:
                print(f"    SKIP (K={num_topics}): {run.name}")
            continue
        
        num_seeds = run.config.get("num_seeds", 0)
        if num_seeds != REQUIRED_NUM_SEEDS:
            skip_reasons["wrong_num_seeds"] += 1
            if debug:
                print(f"    SKIP (seeds={num_seeds}): {run.name}")
            continue
        
        # Deduplicate
        combination = (method_key, num_topics)
        if combination in seen_combinations:
            skip_reasons["duplicate"] += 1
            if debug:
                print(f"    SKIP (dup): {run.name}")
            continue
        seen_combinations.add(combination)
        
        if debug:
            print(f"    MATCH: {run.name} -> {method_key}, K={num_topics}")
        
        # Extract metrics
        metrics = extract_metrics_from_run(run)
        seed_metrics = extract_seed_metrics_from_run(run)
        
        # Record result
        result = {
            "vocab_size": vocab_size,
            "method": method_key,
            "method_display": METHOD_DISPLAY_NAMES.get(method_key, method_key),
            "K": num_topics,
            "num_seeds": num_seeds,
            "run_id": run.id,
            "run_name": run.name,
        }
        result.update(metrics)
        
        # Add seed-level metrics as lists (for statistical analysis)
        for metric, values in seed_metrics.items():
            result[f"{metric}_seeds"] = values
        
        results.append(result)
        method_k_found[method_key].add(num_topics)
    
    # Print skip summary
    if skip_reasons:
        print(f"  Skip summary: name_mismatch={skip_reasons['name_mismatch']}, "
              f"k_not_in_range={skip_reasons['k_not_in_range']}, "
              f"wrong_num_seeds={skip_reasons['wrong_num_seeds']}, "
              f"duplicate={skip_reasons['duplicate']}")
    
    # Find missing combinations
    missing = {}
    for method_key in all_methods:
        found_k = method_k_found.get(method_key, set())
        missing_k = set(K_VALUES) - found_k
        if missing_k:
            missing[method_key] = sorted(missing_k)
    
    return results, missing


def print_completeness_report(all_missing: dict):
    """Print a report of missing runs."""
    print("\n" + "=" * 60)
    print("COMPLETENESS REPORT")
    print("=" * 60)
    
    has_missing = False
    for vocab_size in VOCAB_SIZES:
        missing = all_missing.get(vocab_size, {})
        if missing:
            has_missing = True
            print(f"\nVocab Size {vocab_size}:")
            for method_key, missing_k in sorted(missing.items()):
                display_name = METHOD_DISPLAY_NAMES.get(method_key, method_key)
                print(f"  {display_name}: missing K = {missing_k}")
    
    if not has_missing:
        print("\nAll experiments complete! Each method has runs for K = 25, 50, 75, 100.")
    
    print("=" * 60)


def export_to_csv(all_results: list, output_path: str):
    """Export results to CSV file."""
    if not all_results:
        print("No results to export!")
        return
    
    # Define columns
    columns = [
        "vocab_size", "method", "method_display", "K", "num_seeds",
        "cv_wiki", "llm_rating", "inverted_rbo", "purity", "training_time",
        "run_id", "run_name"
    ]
    
    # Sort results
    all_results.sort(key=lambda x: (x["vocab_size"], x["method"], x["K"]))
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"\nExported {len(all_results)} results to: {output_path}")


def print_summary_table(all_results: list):
    """Print a summary table of results averaged across K values."""
    print("\n" + "=" * 100)
    print("SUMMARY TABLE (averaged across K = 25, 50, 75, 100)")
    print("=" * 100)
    
    # Group by vocab_size and method
    grouped = defaultdict(lambda: defaultdict(list))
    for result in all_results:
        key = (result["vocab_size"], result["method"])
        grouped[key]["cv_wiki"].append(result.get("cv_wiki"))
        grouped[key]["llm_rating"].append(result.get("llm_rating"))
        grouped[key]["inverted_rbo"].append(result.get("inverted_rbo"))
        grouped[key]["purity"].append(result.get("purity"))
    
    # Print header
    header = f"{'Method':<25}"
    for vocab_size in VOCAB_SIZES:
        header += f" | V={vocab_size:>4}"
    print(header)
    print("-" * len(header))
    
    # Print for each metric
    for metric in METRICS:
        print(f"\n{metric.upper()}:")
        all_methods = BASELINE_METHODS + [f"generative_{LLM_MODEL}"]
        for method_key in all_methods:
            display_name = METHOD_DISPLAY_NAMES.get(method_key, method_key)
            row = f"  {display_name:<23}"
            for vocab_size in VOCAB_SIZES:
                values = grouped[(vocab_size, method_key)][metric]
                values = [v for v in values if v is not None]
                if values:
                    avg = sum(values) / len(values)
                    row += f" | {avg:>6.3f}"
                else:
                    row += f" |      -"
            print(row)
    
    print("=" * 100)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Summarize vocab size experiments from wandb")
    parser.add_argument("--debug", action="store_true", help="Enable debug output to see which runs are matched/skipped")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Vocab Size Experiments Summary")
    print("=" * 60)
    print(f"Entity: {settings.wandb_entity}")
    print(f"Vocab sizes: {VOCAB_SIZES}")
    print(f"K values: {K_VALUES}")
    print(f"Required seeds: {REQUIRED_NUM_SEEDS}")
    print(f"Methods: {len(BASELINE_METHODS)} baselines + 1 generative")
    print("=" * 60 + "\n")
    
    # Collect all results
    all_results = []
    all_missing = {}
    
    for vocab_size in VOCAB_SIZES:
        print(f"\n--- Processing vocab_size = {vocab_size} ---")
        results, missing = process_vocab_size_runs(vocab_size, debug=args.debug)
        all_results.extend(results)
        all_missing[vocab_size] = missing
        print(f"  Collected {len(results)} valid runs")
    
    # Print completeness report
    print_completeness_report(all_missing)
    
    # Print summary table
    print_summary_table(all_results)
    
    # Export to CSV
    output_path = os.path.join(os.path.dirname(__file__), "vocab_size_results.csv")
    export_to_csv(all_results, output_path)
    
    # Summary statistics
    print(f"\nTotal runs collected: {len(all_results)}")
    expected = len(VOCAB_SIZES) * (len(BASELINE_METHODS) + 1) * len(K_VALUES)
    print(f"Expected runs: {expected}")
    if len(all_results) == expected:
        print("✓ All experiments complete!")
    else:
        print(f"⚠ Missing {expected - len(all_results)} runs")


if __name__ == "__main__":
    main()

