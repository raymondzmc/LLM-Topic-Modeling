"""
Visualize benchmark results comparing embedding model overhead
with LLM-based processing for topic modeling.
"""

import os
import json
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Color scheme
COLORS = {
    "embedding": "#3498db",  # Blue for embedding models
    "llm": "#e74c3c",        # Red for LLM models
    "llm_small": "#27ae60",  # Green for small LLMs (ERNIE-0.3B)
}


def load_results(input_path: str) -> dict:
    """Load benchmark results from JSON file."""
    with open(input_path, "r") as f:
        return json.load(f)


def create_throughput_comparison(results: list[dict], output_dir: str):
    """Create bar chart comparing document processing throughput."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Separate by type
    embedding_results = [r for r in results if r["model_type"] == "embedding"]
    llm_results = [r for r in results if r["model_type"] == "llm"]
    
    # Sort by throughput
    embedding_results = sorted(embedding_results, key=lambda x: x["docs_per_second"], reverse=True)
    llm_results = sorted(llm_results, key=lambda x: x["docs_per_second"], reverse=True)
    
    all_results = embedding_results + llm_results
    
    # Prepare data
    names = []
    throughputs = []
    colors = []
    
    for r in all_results:
        # Shorten model names for display
        name = r["model_name"]
        if "/" in name:
            name = name.split("/")[-1]
        names.append(name)
        throughputs.append(r["docs_per_second"])
        
        if r["model_type"] == "embedding":
            colors.append(COLORS["embedding"])
        elif "ERNIE" in r["model_name"] or "0.3B" in r["model_name"]:
            colors.append(COLORS["llm_small"])
        else:
            colors.append(COLORS["llm"])
    
    # Create horizontal bar chart
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, throughputs, color=colors, edgecolor="white", linewidth=0.5)
    
    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Documents per Second (higher is better)", fontsize=11)
    ax.set_title("Document Processing Throughput: Embedding Models vs LLMs", fontsize=13, fontweight="bold")
    
    # Add value labels
    for bar, throughput in zip(bars, throughputs):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{throughput:.1f}', va='center', fontsize=9)
    
    # Add legend
    legend_patches = [
        mpatches.Patch(color=COLORS["embedding"], label="Embedding Models (Baselines)"),
        mpatches.Patch(color=COLORS["llm_small"], label="Small LLM (ERNIE-0.3B, Ours)"),
        mpatches.Patch(color=COLORS["llm"], label="Larger LLMs (Ours)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
    
    # Add vertical line for reference
    if embedding_results:
        avg_embedding = np.mean([r["docs_per_second"] for r in embedding_results])
        ax.axvline(x=avg_embedding, color=COLORS["embedding"], linestyle="--", alpha=0.5, linewidth=1)
    
    ax.set_xlim(0, max(throughputs) * 1.15)
    ax.invert_yaxis()  # Top to bottom
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "throughput_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def create_time_per_doc_comparison(results: list[dict], output_dir: str):
    """Create bar chart comparing time per document."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Separate by type
    embedding_results = [r for r in results if r["model_type"] == "embedding"]
    llm_results = [r for r in results if r["model_type"] == "llm"]
    
    # Sort by time (ascending - fastest first)
    embedding_results = sorted(embedding_results, key=lambda x: x["time_per_doc_ms"])
    llm_results = sorted(llm_results, key=lambda x: x["time_per_doc_ms"])
    
    all_results = embedding_results + llm_results
    
    # Prepare data
    names = []
    times = []
    colors = []
    
    for r in all_results:
        name = r["model_name"]
        if "/" in name:
            name = name.split("/")[-1]
        names.append(name)
        times.append(r["time_per_doc_ms"])
        
        if r["model_type"] == "embedding":
            colors.append(COLORS["embedding"])
        elif "ERNIE" in r["model_name"] or "0.3B" in r["model_name"]:
            colors.append(COLORS["llm_small"])
        else:
            colors.append(COLORS["llm"])
    
    # Create horizontal bar chart
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, times, color=colors, edgecolor="white", linewidth=0.5)
    
    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Time per Document (ms, lower is better)", fontsize=11)
    ax.set_title("Document Processing Latency: Embedding Models vs LLMs", fontsize=13, fontweight="bold")
    
    # Add value labels
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + max(times)*0.02, bar.get_y() + bar.get_height()/2,
                f'{t:.2f} ms', va='center', fontsize=9)
    
    # Add legend
    legend_patches = [
        mpatches.Patch(color=COLORS["embedding"], label="Embedding Models (Baselines)"),
        mpatches.Patch(color=COLORS["llm_small"], label="Small LLM (ERNIE-0.3B, Ours)"),
        mpatches.Patch(color=COLORS["llm"], label="Larger LLMs (Ours)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
    
    ax.set_xlim(0, max(times) * 1.2)
    ax.invert_yaxis()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "time_per_doc_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def create_overhead_ratio_plot(results: list[dict], output_dir: str):
    """Create a plot showing overhead ratio of LLMs relative to embedding models."""
    embedding_results = [r for r in results if r["model_type"] == "embedding"]
    llm_results = [r for r in results if r["model_type"] == "llm"]
    
    if not embedding_results or not llm_results:
        print("Skipping overhead ratio plot (need both embedding and LLM results)")
        return
    
    # Use all-mpnet-base-v2 as the reference (ZeroShotTM default)
    reference = None
    for r in embedding_results:
        if "mpnet-base" in r["model_name"].lower():
            reference = r
            break
    
    if reference is None:
        # Use fastest embedding model as reference
        reference = min(embedding_results, key=lambda x: x["time_per_doc_ms"])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Calculate overhead ratios
    models = []
    ratios = []
    colors = []
    
    # Add embedding models
    for r in sorted(embedding_results, key=lambda x: x["time_per_doc_ms"]):
        name = r["model_name"]
        if "/" in name:
            name = name.split("/")[-1]
        models.append(name)
        ratios.append(r["time_per_doc_ms"] / reference["time_per_doc_ms"])
        colors.append(COLORS["embedding"])
    
    # Add LLM models
    for r in sorted(llm_results, key=lambda x: x["time_per_doc_ms"]):
        name = r["model_name"]
        if "/" in name:
            name = name.split("/")[-1]
        models.append(name)
        ratios.append(r["time_per_doc_ms"] / reference["time_per_doc_ms"])
        if "ERNIE" in r["model_name"] or "0.3B" in r["model_name"]:
            colors.append(COLORS["llm_small"])
        else:
            colors.append(COLORS["llm"])
    
    # Create bar chart
    x_pos = np.arange(len(models))
    bars = ax.bar(x_pos, ratios, color=colors, edgecolor="white", linewidth=0.5)
    
    # Add reference line at 1.0
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, linewidth=1, label="Reference (1.0x)")
    
    # Customize
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Overhead Ratio (relative to baseline)", fontsize=11)
    
    ref_name = reference["model_name"]
    if "/" in ref_name:
        ref_name = ref_name.split("/")[-1]
    ax.set_title(f"Computational Overhead Relative to {ref_name}", fontsize=13, fontweight="bold")
    
    # Add value labels
    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{ratio:.2f}x', ha='center', va='bottom', fontsize=8)
    
    # Add legend
    legend_patches = [
        mpatches.Patch(color=COLORS["embedding"], label="Embedding Models (Baselines)"),
        mpatches.Patch(color=COLORS["llm_small"], label="Small LLM (ERNIE-0.3B, Ours)"),
        mpatches.Patch(color=COLORS["llm"], label="Larger LLMs (Ours)"),
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=9)
    
    ax.set_ylim(0, max(ratios) * 1.2)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "overhead_ratio.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def create_model_size_vs_speed_plot(results: list[dict], output_dir: str):
    """Create scatter plot showing model size vs processing speed."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for r in results:
        size = r["model_size_mb"]
        speed = r["docs_per_second"]
        
        if r["model_type"] == "embedding":
            color = COLORS["embedding"]
            marker = "o"
        elif "ERNIE" in r["model_name"] or "0.3B" in r["model_name"]:
            color = COLORS["llm_small"]
            marker = "s"
        else:
            color = COLORS["llm"]
            marker = "s"
        
        ax.scatter(size, speed, c=color, marker=marker, s=150, alpha=0.8, edgecolors="white", linewidth=1)
        
        # Add label
        name = r["model_name"]
        if "/" in name:
            name = name.split("/")[-1]
        ax.annotate(name, (size, speed), textcoords="offset points", xytext=(5, 5), fontsize=8)
    
    ax.set_xlabel("Model Size (MB)", fontsize=11)
    ax.set_ylabel("Documents per Second", fontsize=11)
    ax.set_title("Model Size vs Processing Speed", fontsize=13, fontweight="bold")
    
    # Log scale for x-axis if there's large variation
    sizes = [r["model_size_mb"] for r in results]
    if max(sizes) / min(sizes) > 10:
        ax.set_xscale("log")
    
    # Add legend
    legend_patches = [
        mpatches.Patch(color=COLORS["embedding"], label="Embedding Models (Baselines)"),
        mpatches.Patch(color=COLORS["llm_small"], label="Small LLM (ERNIE-0.3B, Ours)"),
        mpatches.Patch(color=COLORS["llm"], label="Larger LLMs (Ours)"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9)
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "size_vs_speed.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def create_summary_table(results: list[dict], output_dir: str):
    """Create a summary markdown table."""
    embedding_results = [r for r in results if r["model_type"] == "embedding"]
    llm_results = [r for r in results if r["model_type"] == "llm"]
    
    # Find reference (all-mpnet-base-v2 or fastest embedding)
    reference = None
    for r in embedding_results:
        if "mpnet-base" in r["model_name"].lower():
            reference = r
            break
    if reference is None and embedding_results:
        reference = min(embedding_results, key=lambda x: x["time_per_doc_ms"])
    
    lines = [
        "# Embedding Overhead Benchmark Results\n",
        f"Generated: {datetime.now().isoformat()}\n",
        f"Number of documents: {results[0]['num_documents'] if results else 'N/A'}\n",
        f"Device: {results[0].get('gpu_name') or results[0].get('device', 'N/A') if results else 'N/A'}\n",
        "\n## Embedding Models (Used in Baseline Topic Models)\n",
        "| Model | Size (MB) | Docs/sec | ms/doc | Overhead |",
        "|-------|-----------|----------|--------|----------|",
    ]
    
    ref_time = reference["time_per_doc_ms"] if reference else 1.0
    
    for r in sorted(embedding_results, key=lambda x: x["time_per_doc_ms"]):
        name = r["model_name"]
        overhead = r["time_per_doc_ms"] / ref_time
        lines.append(f"| {name} | {r['model_size_mb']:.1f} | {r['docs_per_second']:.1f} | {r['time_per_doc_ms']:.2f} | {overhead:.2f}x |")
    
    lines.extend([
        "\n## LLM Models (Our Approach)\n",
        "| Model | Size (MB) | Docs/sec | ms/doc | Overhead |",
        "|-------|-----------|----------|--------|----------|",
    ])
    
    for r in sorted(llm_results, key=lambda x: x["time_per_doc_ms"]):
        name = r["model_name"]
        overhead = r["time_per_doc_ms"] / ref_time
        lines.append(f"| {name} | {r['model_size_mb']:.1f} | {r['docs_per_second']:.1f} | {r['time_per_doc_ms']:.2f} | {overhead:.2f}x |")
    
    # Add key findings
    if llm_results and embedding_results:
        fastest_embedding = min(embedding_results, key=lambda x: x["time_per_doc_ms"])
        smallest_llm = None
        for r in llm_results:
            if "ERNIE" in r["model_name"] or "0.3B" in r["model_name"]:
                smallest_llm = r
                break
        if smallest_llm is None:
            smallest_llm = min(llm_results, key=lambda x: x["time_per_doc_ms"])
        
        overhead = smallest_llm["time_per_doc_ms"] / reference["time_per_doc_ms"]
        
        lines.extend([
            "\n## Key Findings\n",
            f"- **Reference baseline**: {reference['model_name']} (used in ZeroShotTM/CombinedTM)",
            f"- **Our smallest LLM**: {smallest_llm['model_name']}",
            f"- **Overhead ratio**: {overhead:.2f}x (LLM vs baseline embedding)",
            "",
            "### Interpretation",
            f"- The small ERNIE-0.3B model adds only {overhead:.2f}x overhead compared to Sentence-BERT embeddings",
            f"- This is a modest increase considering the LLM provides richer semantic signals for topic modeling",
            f"- For a dataset of 10,000 documents, processing would take approximately:",
            f"  - Baseline (Sentence-BERT): {reference['time_per_doc_ms'] * 10000 / 1000:.1f} seconds",
            f"  - Our approach (ERNIE-0.3B): {smallest_llm['time_per_doc_ms'] * 10000 / 1000:.1f} seconds",
        ])
    
    output_path = os.path.join(output_dir, "benchmark_summary.md")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument("--input", type=str, default="analysis/benchmark_results.json",
                        help="Path to benchmark results JSON")
    parser.add_argument("--output_dir", type=str, default="analysis/figures",
                        help="Output directory for figures")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Results file not found: {args.input}")
        print("Run the benchmark first: python analysis/benchmark_embedding_overhead.py")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading results from: {args.input}")
    data = load_results(args.input)
    results = data["results"]
    
    print(f"Found {len(results)} benchmark results")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_throughput_comparison(results, args.output_dir)
    create_time_per_doc_comparison(results, args.output_dir)
    create_overhead_ratio_plot(results, args.output_dir)
    create_model_size_vs_speed_plot(results, args.output_dir)
    create_summary_table(results, args.output_dir)
    
    print(f"\nAll figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

