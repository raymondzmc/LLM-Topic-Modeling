"""Visualize temperature ablation experiments.

Creates a multi-axis line chart showing how different temperature values
affect topic model metrics (CV, LLM Rating, I-RBO, Purity).

Dataset: 20_newsgroups
Run pattern: generative_ERNIE-4.5-0.3B-PT_K{k}_temp{t}
  - k = 25, 50, 75, 100
  - t = 0.5, 1.0, 1.5, ..., 9.0
  - Exception: t=3.0 has no _temp suffix (default temperature)
"""

import sys
import os
import re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from settings import settings

# Configuration
DATASETS = ["20_newsgroups", "tweet_topic", "stackoverflow"]
LLM_MODEL = "ERNIE-4.5-0.3B-PT"
K_VALUES = [25, 50, 75, 100]
REQUIRED_NUM_SEEDS = 5
DEFAULT_TEMPERATURE = 3.0

# Metrics to extract
METRICS = ["cv_wiki", "llm_rating", "inverted_rbo", "purity"]
METRIC_DISPLAY_NAMES = {
    "cv_wiki": r"$C_V$",
    "llm_rating": "LLM",
    "inverted_rbo": "I-RBO",
    "purity": "Purity",
}

# Colors for each metric line
METRIC_COLORS = {
    "cv_wiki": "#E63946",       # Red
    "llm_rating": "#F4A261",    # Orange
    "inverted_rbo": "#2A9D8F",  # Teal
    "purity": "#264653",        # Dark blue
}


def fetch_runs(dataset: str) -> list:
    """Fetch all finished runs for the dataset from wandb."""
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


def parse_temperature_from_run(run) -> float | None:
    """Extract temperature value from a run.
    
    Returns:
        Temperature value, or None if not a matching run.
    """
    model = run.config.get("model", "")
    if model != "generative":
        return None
    
    run_name = run.name
    expected_prefix = f"generative_{LLM_MODEL}_K"
    
    if not run_name.startswith(expected_prefix):
        return None
    
    # Extract the part after the prefix
    suffix = run_name[len(expected_prefix):]
    
    # Check for temperature suffix: {k}_temp{t} or just {k} (default temp)
    temp_match = re.match(r"(\d+)_temp([\d.]+)$", suffix)
    if temp_match:
        return float(temp_match.group(2))
    
    # Check for default temperature (no suffix, just K number)
    if suffix.isdigit():
        return DEFAULT_TEMPERATURE
    
    # Not a temperature ablation run (could be other ablation like _CE, _bow-target, etc.)
    return None


def extract_metrics_from_run(run) -> dict:
    """Extract the required metrics from a run's summary."""
    metrics = {}
    for metric in METRICS:
        key = f"avg/{metric}"
        value = run.summary.get(key)
        if value is not None:
            metrics[metric] = value
    return metrics


def build_temperature_data(runs: list) -> dict:
    """Build aggregated metrics data by temperature.
    
    Returns:
        Dict: {temperature: {metric: [values]}}
    """
    # Collect data: {temperature: {k: {metric: value}}}
    temp_k_data = defaultdict(lambda: defaultdict(dict))
    
    # Track seen (temperature, K) combinations to take only the latest run
    seen_combinations = set()
    
    for run in runs:
        # Parse temperature
        temperature = parse_temperature_from_run(run)
        if temperature is None:
            continue
        
        # Get K value
        num_topics = run.config.get("num_topics")
        if num_topics not in K_VALUES:
            continue
        
        # Check if run has the required number of seeds
        num_seeds = run.config.get("num_seeds", 0)
        if num_seeds != REQUIRED_NUM_SEEDS:
            continue
        
        # Deduplicate: take only the first (latest) run for each (temperature, K)
        combination = (temperature, num_topics)
        if combination in seen_combinations:
            continue
        seen_combinations.add(combination)
        
        # Extract metrics
        metrics = extract_metrics_from_run(run)
        if metrics:
            temp_k_data[temperature][num_topics] = metrics
    
    # Aggregate by temperature (average over K values)
    temp_metrics = {}
    for temperature in sorted(temp_k_data.keys()):
        k_data = temp_k_data[temperature]
        
        # Collect all values for each metric across K values
        metric_values = defaultdict(list)
        for k in K_VALUES:
            if k in k_data:
                for metric in METRICS:
                    if metric in k_data[k]:
                        metric_values[metric].append(k_data[k][metric])
        
        # Average
        temp_metrics[temperature] = {
            metric: np.mean(values) if values else None
            for metric, values in metric_values.items()
        }
        
        # Debug output
        n_k = len(k_data)
        print(f"  Temperature {temperature}: {n_k} K values")
    
    return temp_metrics


def print_data_table(temp_metrics: dict):
    """Print a formatted table of the temperature data."""
    print("\n" + "=" * 70)
    print("TEMPERATURE ABLATION DATA")
    print("=" * 70)
    
    # Header
    header = f"{'Temp':>6}"
    for metric in METRICS:
        header += f" | {METRIC_DISPLAY_NAMES[metric]:>10}"
    print(header)
    print("-" * 70)
    
    # Data rows
    for temp in sorted(temp_metrics.keys()):
        row = f"{temp:>6.1f}"
        for metric in METRICS:
            val = temp_metrics[temp].get(metric)
            if val is not None:
                row += f" | {val:>10.4f}"
            else:
                row += f" | {'-':>10}"
        print(row)
    
    print("=" * 70)


def normalize_data(data: list) -> list:
    """Normalize data to 0-1 range for trend visualization."""
    valid = [v for v in data if v is not None]
    if not valid:
        return data
    min_val, max_val = min(valid), max(valid)
    range_val = max_val - min_val if max_val != min_val else 1
    return [(v - min_val) / range_val if v is not None else None for v in data]


def plot_temperature_ablation(temp_metrics: dict, output_path: str):
    """Create a clean, minimal line chart for temperature ablation.
    
    Designed for ACL double-column format:
    - No Y-axes (trend-focused)
    - Large fonts for readability
    - Thick lines and markers
    - Clean legend
    """
    temperatures = sorted(temp_metrics.keys())
    
    # Extract and normalize data for each metric
    metric_data = {}
    for metric in METRICS:
        raw_data = [temp_metrics[t].get(metric) for t in temperatures]
        metric_data[metric] = normalize_data(raw_data)
    
    # Create figure - compact size for double-column
    fig, ax = plt.subplots(figsize=(5, 3))
    
    # Plot order for legend
    plot_order = ["cv_wiki", "llm_rating", "inverted_rbo", "purity"]
    
    lines = []
    labels = []
    
    for metric in plot_order:
        color = METRIC_COLORS[metric]
        data = metric_data[metric]
        
        # Filter out None values
        valid_temps = [t for t, v in zip(temperatures, data) if v is not None]
        valid_data = [v for v in data if v is not None]
        
        if valid_data:
            line, = ax.plot(valid_temps, valid_data, 'o-', color=color, 
                           linewidth=2.5, markersize=7, 
                           label=METRIC_DISPLAY_NAMES[metric],
                           markeredgecolor='white', markeredgewidth=0.8)
            lines.append(line)
            labels.append(METRIC_DISPLAY_NAMES[metric])
    
    # Remove all spines except bottom
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['bottom'].set_color('#333333')
    
    # Remove Y-axis ticks and labels
    ax.set_yticks([])
    ax.set_ylabel('')
    
    # X-axis styling
    ax.set_xlabel(r'$\tau$', fontsize=14, fontweight='medium', labelpad=6)
    ax.set_xticks(temperatures)
    ax.set_xticklabels([f'{t:.1f}' if t != int(t) else f'{int(t)}' for t in temperatures], 
                       fontsize=11)
    ax.tick_params(axis='x', length=4, width=1.2, colors='#333333')
    ax.set_xlim(min(temperatures) - 0.3, max(temperatures) + 0.3)
    
    # Y limits with padding for normalized data
    ax.set_ylim(-0.08, 1.12)
    
    # Subtle horizontal grid for reference
    ax.axhline(y=0, color='#e0e0e0', linewidth=0.8, zorder=0)
    ax.axhline(y=0.5, color='#e0e0e0', linewidth=0.8, linestyle='--', zorder=0)
    ax.axhline(y=1, color='#e0e0e0', linewidth=0.8, zorder=0)
    
    # Legend - below plot, horizontal
    legend = ax.legend(lines, labels, loc='upper center', 
                       bbox_to_anchor=(0.5, -0.28), ncol=4, fontsize=10,
                       frameon=False, handlelength=1.5, handletextpad=0.4,
                       columnspacing=1.0)
    
    # Clean background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', pad_inches=0.1)
    print(f"\nFigure saved to: {output_path}")
    
    plt.close()


def main():
    print("=" * 60)
    print("Temperature Ablation Visualization")
    print("=" * 60)
    print(f"Datasets: {DATASETS}")
    print(f"LLM Model: {LLM_MODEL}")
    print(f"K values: {K_VALUES}")
    print(f"Default temperature: {DEFAULT_TEMPERATURE}")
    print("=" * 60 + "\n")
    
    for dataset in DATASETS:
        print("\n" + "-" * 60)
        print(f"Processing: {dataset}")
        print("-" * 60)
        
        # Fetch runs
        runs = fetch_runs(dataset)
        if not runs:
            print(f"No runs found for {dataset}!")
            continue
        
        # Build temperature data
        print("\nProcessing runs by temperature...")
        temp_metrics = build_temperature_data(runs)
        
        if not temp_metrics:
            print(f"No temperature ablation data found for {dataset}!")
            continue
        
        # Print data table
        print_data_table(temp_metrics)
        
        # Plot
        output_path = os.path.join(
            os.path.dirname(__file__), 
            "figures", 
            f"temperature_ablation_{dataset}.png"
        )
        plot_temperature_ablation(temp_metrics, output_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

