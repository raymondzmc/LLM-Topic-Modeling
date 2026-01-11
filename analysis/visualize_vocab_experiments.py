"""Visualize vocab size experiments.

Creates plots showing how different vocabulary sizes affect topic model metrics
across all methods (baselines + generative ERNIE).

Reads from: vocab_size_results.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Configuration
VOCAB_SIZES = [500, 1000, 2000, 4000]
K_VALUES = [25, 50, 75, 100]

# Metrics
METRICS = ["cv_wiki", "llm_rating", "inverted_rbo", "purity"]
METRIC_DISPLAY_NAMES = {
    "cv_wiki": r"$C_V$ (Coherence)",
    "llm_rating": "LLM Rating",
    "inverted_rbo": "I-RBO (Diversity)",
    "purity": "Purity (Clustering)",
}
METRIC_SHORT_NAMES = {
    "cv_wiki": r"$C_V$",
    "llm_rating": "LLM",
    "inverted_rbo": "I-RBO",
    "purity": "Purity",
}

# Method configuration
BASELINE_METHODS = ["lda", "prodlda", "zeroshot", "combined", "etm", "bertopic", "ecrtm", "fastopic"]
GENERATIVE_METHOD = "generative_ERNIE-4.5-0.3B-PT"

METHOD_DISPLAY_NAMES = {
    "lda": "LDA",
    "prodlda": "ProdLDA",
    "zeroshot": "ZeroShotTM",
    "combined": "CombinedTM",
    "etm": "ETM",
    "bertopic": "BERTopic",
    "ecrtm": "ECRTM",
    "fastopic": "FASTopic",
    GENERATIVE_METHOD: "Ours (ERNIE)",
}

# Colors - distinct palette for all methods
METHOD_COLORS = {
    "lda": "#8B8B8B",           # Gray
    "prodlda": "#A0A0A0",       # Light gray
    "zeroshot": "#7FB3D5",      # Light blue
    "combined": "#5499C7",      # Blue
    "etm": "#82E0AA",           # Light green
    "bertopic": "#F7DC6F",      # Yellow
    "ecrtm": "#E59866",         # Orange
    "fastopic": "#C39BD3",      # Purple
    GENERATIVE_METHOD: "#E74C3C",  # Red (stands out)
}

# Metric colors for multi-metric plots
METRIC_COLORS = {
    "cv_wiki": "#E63946",       # Red
    "llm_rating": "#F4A261",    # Orange
    "inverted_rbo": "#2A9D8F",  # Teal
    "purity": "#264653",        # Dark blue
}


def load_data(csv_path: str) -> dict:
    """Load data from CSV file.
    
    Returns:
        Dict: {method: {vocab_size: {metric: [values_per_k]}}}
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"]
            vocab_size = int(row["vocab_size"])
            
            for metric in METRICS:
                value = row.get(metric)
                if value:
                    data[method][vocab_size][metric].append(float(value))
    
    return data


def aggregate_over_k(data: dict) -> dict:
    """Aggregate metrics over K values.
    
    Returns:
        Dict: {method: {vocab_size: {metric: mean_value}}}
    """
    aggregated = {}
    for method in data:
        aggregated[method] = {}
        for vocab_size in data[method]:
            aggregated[method][vocab_size] = {}
            for metric in METRICS:
                values = data[method][vocab_size][metric]
                if values:
                    aggregated[method][vocab_size][metric] = np.mean(values)
    return aggregated


def plot_all_methods_per_metric(data: dict, output_dir: str):
    """Create one plot per metric showing all methods vs vocab size."""
    
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(7, 5))
        
        # Plot baselines first (thinner, muted)
        for method in BASELINE_METHODS:
            if method not in data:
                continue
            
            x_vals = []
            y_vals = []
            for vocab_size in VOCAB_SIZES:
                if vocab_size in data[method] and metric in data[method][vocab_size]:
                    x_vals.append(vocab_size)
                    y_vals.append(data[method][vocab_size][metric])
            
            if y_vals:
                ax.plot(x_vals, y_vals, 'o-', 
                       color=METHOD_COLORS[method], 
                       linewidth=1.5, markersize=5, alpha=0.7,
                       label=METHOD_DISPLAY_NAMES[method])
        
        # Plot generative method (thicker, stands out)
        if GENERATIVE_METHOD in data:
            x_vals = []
            y_vals = []
            for vocab_size in VOCAB_SIZES:
                if vocab_size in data[GENERATIVE_METHOD] and metric in data[GENERATIVE_METHOD][vocab_size]:
                    x_vals.append(vocab_size)
                    y_vals.append(data[GENERATIVE_METHOD][vocab_size][metric])
            
            if y_vals:
                ax.plot(x_vals, y_vals, 'o-', 
                       color=METHOD_COLORS[GENERATIVE_METHOD], 
                       linewidth=3, markersize=8,
                       label=METHOD_DISPLAY_NAMES[GENERATIVE_METHOD],
                       zorder=10)
        
        # Styling
        ax.set_xlabel("Vocabulary Size", fontsize=12)
        ax.set_ylabel(METRIC_DISPLAY_NAMES[metric], fontsize=12)
        ax.set_xticks(VOCAB_SIZES)
        ax.set_xticklabels([str(v) for v in VOCAB_SIZES])
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Legend
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
        
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(output_dir, f"vocab_size_{metric}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
        plt.close()


def plot_generative_vs_baselines(data: dict, output_dir: str):
    """Create a 2x2 grid comparing generative vs average baseline for each metric."""
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        
        # Calculate baseline average
        baseline_means = {v: [] for v in VOCAB_SIZES}
        for method in BASELINE_METHODS:
            if method not in data:
                continue
            for vocab_size in VOCAB_SIZES:
                if vocab_size in data[method] and metric in data[method][vocab_size]:
                    baseline_means[vocab_size].append(data[method][vocab_size][metric])
        
        baseline_avg = []
        baseline_std = []
        for vocab_size in VOCAB_SIZES:
            if baseline_means[vocab_size]:
                baseline_avg.append(np.mean(baseline_means[vocab_size]))
                baseline_std.append(np.std(baseline_means[vocab_size]))
            else:
                baseline_avg.append(None)
                baseline_std.append(None)
        
        # Get generative values
        gen_vals = []
        for vocab_size in VOCAB_SIZES:
            if GENERATIVE_METHOD in data and vocab_size in data[GENERATIVE_METHOD]:
                if metric in data[GENERATIVE_METHOD][vocab_size]:
                    gen_vals.append(data[GENERATIVE_METHOD][vocab_size][metric])
                else:
                    gen_vals.append(None)
            else:
                gen_vals.append(None)
        
        # Plot baseline average with error band
        valid_vocab = [v for v, avg in zip(VOCAB_SIZES, baseline_avg) if avg is not None]
        valid_avg = [avg for avg in baseline_avg if avg is not None]
        valid_std = [std for std, avg in zip(baseline_std, baseline_avg) if avg is not None]
        
        if valid_avg:
            ax.fill_between(valid_vocab, 
                           [a - s for a, s in zip(valid_avg, valid_std)],
                           [a + s for a, s in zip(valid_avg, valid_std)],
                           alpha=0.2, color='#7F8C8D')
            ax.plot(valid_vocab, valid_avg, 'o-', 
                   color='#7F8C8D', linewidth=2, markersize=6,
                   label='Baselines (avg ± std)')
        
        # Plot generative
        valid_vocab_gen = [v for v, val in zip(VOCAB_SIZES, gen_vals) if val is not None]
        valid_gen = [val for val in gen_vals if val is not None]
        
        if valid_gen:
            ax.plot(valid_vocab_gen, valid_gen, 'o-', 
                   color='#E74C3C', linewidth=3, markersize=8,
                   label='Ours (ERNIE)')
        
        # Styling
        ax.set_xlabel("Vocabulary Size", fontsize=11)
        ax.set_ylabel(METRIC_SHORT_NAMES[metric], fontsize=12, fontweight='medium')
        ax.set_xticks(VOCAB_SIZES)
        ax.set_xticklabels([str(v) for v in VOCAB_SIZES])
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if idx == 0:
            ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, "vocab_size_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def plot_normalized_trend(data: dict, output_dir: str):
    """Create a single plot with normalized metrics to show trends.
    
    Similar style to temperature ablation plot - shows all metrics for the generative
    method normalized to 0-1 range.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Normalize data function
    def normalize(values):
        valid = [v for v in values if v is not None]
        if not valid or len(set(valid)) <= 1:
            return values
        min_val, max_val = min(valid), max(valid)
        return [(v - min_val) / (max_val - min_val) if v is not None else None for v in values]
    
    # Get generative method data
    if GENERATIVE_METHOD not in data:
        print("Generative method not found in data!")
        return
    
    for metric in METRICS:
        values = []
        for vocab_size in VOCAB_SIZES:
            if vocab_size in data[GENERATIVE_METHOD] and metric in data[GENERATIVE_METHOD][vocab_size]:
                values.append(data[GENERATIVE_METHOD][vocab_size][metric])
            else:
                values.append(None)
        
        # Normalize
        norm_values = normalize(values)
        
        # Filter out None values for plotting
        valid_vocab = [v for v, val in zip(VOCAB_SIZES, norm_values) if val is not None]
        valid_values = [val for val in norm_values if val is not None]
        
        if valid_values:
            ax.plot(valid_vocab, valid_values, 'o-', 
                   color=METRIC_COLORS[metric],
                   linewidth=2.5, markersize=7,
                   label=METRIC_SHORT_NAMES[metric],
                   markeredgecolor='white', markeredgewidth=0.8)
    
    # Styling (similar to temperature ablation)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['bottom'].set_color('#333333')
    
    # Remove Y-axis
    ax.set_yticks([])
    ax.set_ylabel('')
    
    # X-axis
    ax.set_xlabel("Vocabulary Size", fontsize=12, fontweight='medium')
    ax.set_xticks(VOCAB_SIZES)
    ax.set_xticklabels([str(v) for v in VOCAB_SIZES], fontsize=11)
    ax.tick_params(axis='x', length=4, width=1.2, colors='#333333')
    ax.set_xlim(VOCAB_SIZES[0] - 200, VOCAB_SIZES[-1] + 200)
    
    # Y limits
    ax.set_ylim(-0.08, 1.12)
    
    # Reference lines
    ax.axhline(y=0, color='#e0e0e0', linewidth=0.8, zorder=0)
    ax.axhline(y=0.5, color='#e0e0e0', linewidth=0.8, linestyle='--', zorder=0)
    ax.axhline(y=1, color='#e0e0e0', linewidth=0.8, zorder=0)
    
    # Legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, 
             fontsize=10, frameon=False, handlelength=1.5, columnspacing=1.0)
    
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, "vocab_size_trend.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    print(f"Saved: {output_path}")
    plt.close()


def plot_bar_comparison(data: dict, output_dir: str):
    """Create grouped bar chart comparing methods at each vocab size."""
    
    all_methods = BASELINE_METHODS + [GENERATIVE_METHOD]
    n_methods = len(all_methods)
    n_vocab = len(VOCAB_SIZES)
    
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        x = np.arange(n_vocab)
        width = 0.08
        
        for i, method in enumerate(all_methods):
            if method not in data:
                continue
            
            values = []
            for vocab_size in VOCAB_SIZES:
                if vocab_size in data[method] and metric in data[method][vocab_size]:
                    values.append(data[method][vocab_size][metric])
                else:
                    values.append(0)
            
            offset = (i - n_methods / 2) * width
            color = METHOD_COLORS[method]
            alpha = 1.0 if method == GENERATIVE_METHOD else 0.7
            edgecolor = 'black' if method == GENERATIVE_METHOD else 'none'
            linewidth = 1.5 if method == GENERATIVE_METHOD else 0
            
            ax.bar(x + offset, values, width, 
                  label=METHOD_DISPLAY_NAMES[method],
                  color=color, alpha=alpha,
                  edgecolor=edgecolor, linewidth=linewidth)
        
        ax.set_xlabel("Vocabulary Size", fontsize=12)
        ax.set_ylabel(METRIC_DISPLAY_NAMES[metric], fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in VOCAB_SIZES])
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f"vocab_size_bar_{metric}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
        plt.close()


def print_summary_table(data: dict):
    """Print a formatted summary table."""
    print("\n" + "=" * 80)
    print("VOCAB SIZE EXPERIMENT SUMMARY (averaged over K=25,50,75,100)")
    print("=" * 80)
    
    all_methods = BASELINE_METHODS + [GENERATIVE_METHOD]
    
    for metric in METRICS:
        print(f"\n{METRIC_DISPLAY_NAMES[metric]}:")
        header = f"{'Method':<20}"
        for v in VOCAB_SIZES:
            header += f" | V={v:>4}"
        print(header)
        print("-" * len(header))
        
        for method in all_methods:
            if method not in data:
                continue
            display = METHOD_DISPLAY_NAMES[method]
            row = f"{display:<20}"
            for vocab_size in VOCAB_SIZES:
                if vocab_size in data[method] and metric in data[method][vocab_size]:
                    val = data[method][vocab_size][metric]
                    row += f" | {val:>6.3f}"
                else:
                    row += f" |      -"
            print(row)
    
    print("=" * 80)


def main():
    print("=" * 60)
    print("Vocab Size Experiment Visualization")
    print("=" * 60)
    
    # Load data
    csv_path = os.path.join(os.path.dirname(__file__), "vocab_size_results.csv")
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        print("Run summarize_vocab_experiments.py first!")
        return
    
    print(f"Loading data from: {csv_path}")
    raw_data = load_data(csv_path)
    print(f"Loaded {len(raw_data)} methods")
    
    # Aggregate over K values
    data = aggregate_over_k(raw_data)
    
    # Print summary table
    print_summary_table(data)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots
    print("\nGenerating plots...")
    
    print("\n1. All methods per metric (line plots)...")
    plot_all_methods_per_metric(data, output_dir)
    
    print("\n2. Generative vs baselines comparison (2x2 grid)...")
    plot_generative_vs_baselines(data, output_dir)
    
    print("\n3. Normalized trend plot (generative method)...")
    plot_normalized_trend(data, output_dir)
    
    print("\n4. Grouped bar charts...")
    plot_bar_comparison(data, output_dir)
    
    print("\nDone! All figures saved to:", output_dir)


if __name__ == "__main__":
    main()

