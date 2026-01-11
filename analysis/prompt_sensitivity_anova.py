"""Two-Way ANOVA Analysis for Prompt Sensitivity Experiments.

Performs a two-way ANOVA (Prompt × Seed) to test whether prompt wording
significantly affects topic model metrics.

Factors:
- Factor A (Prompt): 5 levels (variant_1 through variant_5)
- Factor B (Seed): 5 levels (seeds 0-4)

Dependent Variables:
- cv_wiki: Topic coherence (Wikipedia)
- llm_rating: LLM-based topic quality rating
- inverted_rbo: Topic diversity (Inverted RBO)
- purity: Clustering purity

Usage:
    python analysis/prompt_sensitivity_anova.py
"""

import sys
import os
import re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
import numpy as np
import pandas as pd
from collections import defaultdict
from settings import settings
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


# Configuration
WANDB_PROJECT = "prompt_sensitivity"
LLM_MODEL = "ERNIE-4.5-0.3B-PT"
NUM_SEEDS = 5
PROMPT_VARIANTS = ["variant_1", "variant_2", "variant_3", "variant_4", "variant_5"]

# Metrics to analyze
METRICS = ["cv_wiki", "llm_rating", "inverted_rbo", "purity"]
METRIC_DISPLAY_NAMES = {
    "cv_wiki": "C_V (Wiki)",
    "llm_rating": "LLM Rating",
    "inverted_rbo": "I-RBO (Diversity)",
    "purity": "Purity",
}


def fetch_runs() -> list:
    """Fetch all finished runs from the prompt_sensitivity wandb project."""
    api = wandb.Api()
    project_path = f"{settings.wandb_entity}/{WANDB_PROJECT}"
    
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


def get_generative_runs_by_variant(runs: list) -> dict:
    """Group generative ERNIE runs by prompt variant based on creation order.
    
    Since runs are returned in reverse chronological order (newest first),
    and the prompt_sensitivity.sh script runs variants in order (variant_1 first),
    we reverse the order to map: oldest run -> variant_1, newest run -> variant_5.
    
    Returns:
        Dict: {variant: run}
    """
    # Filter for generative ERNIE runs with K=50
    target_name = f"generative_{LLM_MODEL}_K50"
    generative_runs = [
        r for r in runs 
        if r.config.get("model") == "generative" 
        and r.name == target_name
        and r.config.get("num_seeds") == NUM_SEEDS
    ]
    
    # Sort by created_at (oldest first) to match variant order
    generative_runs.sort(key=lambda r: r.created_at)
    
    # Map to variants (variant_1 is the oldest run)
    variant_to_run = {}
    for i, run in enumerate(generative_runs):
        if i < len(PROMPT_VARIANTS):
            variant = PROMPT_VARIANTS[i]
            variant_to_run[variant] = run
    
    return variant_to_run


def get_baseline_runs_by_variant(runs: list, model_type: str) -> dict:
    """Group baseline model runs by prompt variant based on creation order.
    
    Returns:
        Dict: {variant: run}
    """
    # Filter for the specific model type with K=50
    target_runs = [
        r for r in runs 
        if r.config.get("model") == model_type 
        and r.config.get("num_topics") == 50
        and r.config.get("num_seeds") == NUM_SEEDS
    ]
    
    # Sort by created_at (oldest first) to match variant order
    target_runs.sort(key=lambda r: r.created_at)
    
    # Map to variants
    variant_to_run = {}
    for i, run in enumerate(target_runs):
        if i < len(PROMPT_VARIANTS):
            variant = PROMPT_VARIANTS[i]
            variant_to_run[variant] = run
    
    return variant_to_run


def extract_per_seed_metrics(run) -> dict:
    """Extract per-seed metrics from a run's summary.
    
    Returns:
        Dict: {seed: {metric: value}}
    """
    seed_metrics = {}
    
    for seed in range(NUM_SEEDS):
        metrics = {}
        for metric in METRICS:
            key = f"seed_{seed}/{metric}"
            value = run.summary.get(key)
            if value is not None:
                metrics[metric] = value
        
        if metrics:
            seed_metrics[seed] = metrics
    
    return seed_metrics


def build_dataframe(runs: list) -> pd.DataFrame:
    """Build a DataFrame from wandb runs for ANOVA analysis.
    
    Uses run creation order to infer prompt variant (oldest = variant_1).
    
    Returns:
        DataFrame with columns: model, prompt, seed, cv_wiki, llm_rating, inverted_rbo, purity
    """
    records = []
    
    # Process generative ERNIE runs
    print("\n  Processing generative runs...")
    generative_variants = get_generative_runs_by_variant(runs)
    print(f"    Found {len(generative_variants)} variants: {list(generative_variants.keys())}")
    
    for variant, run in generative_variants.items():
        seed_metrics = extract_per_seed_metrics(run)
        print(f"    {variant}: {len(seed_metrics)} seeds")
        
        for seed, metrics in seed_metrics.items():
            record = {
                "model": "generative",
                "prompt": variant,
                "seed": seed,
            }
            record.update(metrics)
            records.append(record)
    
    # Process baseline models (zeroshot, prodlda, fastopic)
    for model_type in ["zeroshot", "prodlda", "fastopic"]:
        print(f"\n  Processing {model_type} runs...")
        baseline_variants = get_baseline_runs_by_variant(runs, model_type)
        print(f"    Found {len(baseline_variants)} variants: {list(baseline_variants.keys())}")
        
        for variant, run in baseline_variants.items():
            seed_metrics = extract_per_seed_metrics(run)
            print(f"    {variant}: {len(seed_metrics)} seeds")
            
            for seed, metrics in seed_metrics.items():
                record = {
                    "model": model_type,
                    "prompt": variant,
                    "seed": seed,
                }
                record.update(metrics)
                records.append(record)
    
    df = pd.DataFrame(records)
    return df


def compute_partial_eta_squared(anova_table: pd.DataFrame, effect: str) -> float:
    """Compute partial eta-squared effect size.
    
    Partial η² = SS_effect / (SS_effect + SS_residual)
    """
    ss_effect = anova_table.loc[effect, "sum_sq"]
    ss_residual = anova_table.loc["Residual", "sum_sq"]
    return ss_effect / (ss_effect + ss_residual)


def interpret_effect_size(eta_sq: float) -> str:
    """Interpret partial eta-squared effect size (Cohen's guidelines)."""
    if eta_sq < 0.01:
        return "negligible"
    elif eta_sq < 0.06:
        return "small"
    elif eta_sq < 0.14:
        return "medium"
    else:
        return "large"


def run_two_way_anova(df: pd.DataFrame, metric: str, include_interaction: bool = False) -> dict:
    """Run two-way ANOVA for a single metric.
    
    For designs without replication (1 obs per cell), use additive model only.
    The interaction term serves as the error term in unreplicated designs.
    
    Args:
        df: DataFrame with columns prompt, seed, and metric
        metric: Name of the dependent variable column
        include_interaction: Whether to include interaction term (requires replication)
    
    Returns:
        Dict with ANOVA results
    """
    # Check if we have replication (multiple obs per cell)
    cell_counts = df.groupby(["prompt", "seed"]).size()
    has_replication = (cell_counts > 1).any()
    
    if include_interaction and has_replication:
        # Full model with interaction
        formula = f"{metric} ~ C(prompt) + C(seed) + C(prompt):C(seed)"
    else:
        # Additive model (interaction used as error term)
        formula = f"{metric} ~ C(prompt) + C(seed)"
    
    model = ols(formula, data=df).fit()
    
    # Get ANOVA table
    anova_table = anova_lm(model, typ=2)
    
    # Extract results
    results = {
        "metric": metric,
        "n_observations": len(df),
        "has_replication": has_replication,
        "model_type": "full" if (include_interaction and has_replication) else "additive",
    }
    
    # Main effect of Prompt
    results["prompt_F"] = anova_table.loc["C(prompt)", "F"]
    results["prompt_p"] = anova_table.loc["C(prompt)", "PR(>F)"]
    results["prompt_eta_sq"] = compute_partial_eta_squared(anova_table, "C(prompt)")
    
    # Main effect of Seed
    results["seed_F"] = anova_table.loc["C(seed)", "F"]
    results["seed_p"] = anova_table.loc["C(seed)", "PR(>F)"]
    results["seed_eta_sq"] = compute_partial_eta_squared(anova_table, "C(seed)")
    
    # Interaction effect (only if included)
    if "C(prompt):C(seed)" in anova_table.index:
        results["interaction_F"] = anova_table.loc["C(prompt):C(seed)", "F"]
        results["interaction_p"] = anova_table.loc["C(prompt):C(seed)", "PR(>F)"]
        results["interaction_eta_sq"] = compute_partial_eta_squared(anova_table, "C(prompt):C(seed)")
    else:
        results["interaction_F"] = None
        results["interaction_p"] = None
        results["interaction_eta_sq"] = None
    
    # Store full table for detailed output
    results["anova_table"] = anova_table
    
    return results


def print_anova_results(results: dict):
    """Print formatted ANOVA results for a single metric."""
    metric = results["metric"]
    display_name = METRIC_DISPLAY_NAMES.get(metric, metric)
    
    print(f"\n{'='*70}")
    print(f"TWO-WAY ANOVA: {display_name}")
    print(f"{'='*70}")
    print(f"N = {results['n_observations']} observations")
    print(f"Model: {results['model_type']} (interaction {'included' if results['interaction_F'] is not None else 'used as error term'})")
    print()
    
    # Full ANOVA table
    print("ANOVA Table:")
    print("-" * 70)
    print(results["anova_table"].to_string())
    print()
    
    # Summary of effects
    print("Summary of Effects:")
    print("-" * 70)
    
    # Prompt effect
    sig_prompt = "***" if results["prompt_p"] < 0.001 else "**" if results["prompt_p"] < 0.01 else "*" if results["prompt_p"] < 0.05 else ""
    effect_prompt = interpret_effect_size(results["prompt_eta_sq"])
    print(f"  Prompt (main effect):      F = {results['prompt_F']:>8.3f}, p = {results['prompt_p']:.4f}{sig_prompt}")
    print(f"                             η²p = {results['prompt_eta_sq']:.4f} ({effect_prompt})")
    
    # Seed effect
    sig_seed = "***" if results["seed_p"] < 0.001 else "**" if results["seed_p"] < 0.01 else "*" if results["seed_p"] < 0.05 else ""
    effect_seed = interpret_effect_size(results["seed_eta_sq"])
    print(f"  Seed (main effect):        F = {results['seed_F']:>8.3f}, p = {results['seed_p']:.4f}{sig_seed}")
    print(f"                             η²p = {results['seed_eta_sq']:.4f} ({effect_seed})")
    
    # Interaction (only if computed)
    if results["interaction_F"] is not None:
        sig_int = "***" if results["interaction_p"] < 0.001 else "**" if results["interaction_p"] < 0.01 else "*" if results["interaction_p"] < 0.05 else ""
        effect_int = interpret_effect_size(results["interaction_eta_sq"])
        print(f"  Prompt × Seed (interaction): F = {results['interaction_F']:>8.3f}, p = {results['interaction_p']:.4f}{sig_int}")
        print(f"                             η²p = {results['interaction_eta_sq']:.4f} ({effect_int})")
    else:
        print(f"  Prompt × Seed (interaction): Not estimated (no replication; used as error term)")
    
    print()
    print("Significance: * p < 0.05, ** p < 0.01, *** p < 0.001")


def print_descriptive_stats(df: pd.DataFrame, metric: str):
    """Print descriptive statistics by prompt variant."""
    display_name = METRIC_DISPLAY_NAMES.get(metric, metric)
    
    print(f"\nDescriptive Statistics for {display_name}:")
    print("-" * 50)
    
    # Group by prompt
    grouped = df.groupby("prompt")[metric].agg(["mean", "std", "count"])
    grouped.columns = ["Mean", "Std", "N"]
    print(grouped.to_string())
    
    # Overall
    print(f"\nOverall: Mean = {df[metric].mean():.4f}, Std = {df[metric].std():.4f}")


def print_summary_table(all_results: list):
    """Print a compact summary table of all ANOVA results."""
    print("\n" + "=" * 90)
    print("SUMMARY TABLE: Two-Way ANOVA Results (Prompt × Seed)")
    print("=" * 90)
    
    # Check if any results have interaction
    has_interaction = any(r["interaction_F"] is not None for r in all_results)
    
    # Header
    if has_interaction:
        print(f"{'Metric':<20} | {'Prompt Effect':<25} | {'Seed Effect':<25} | {'Interaction':<15}")
        print(f"{'':<20} | {'F':>8} {'p':>8} {'η²p':>6} | {'F':>8} {'p':>8} {'η²p':>6} | {'F':>8} {'p':>6}")
    else:
        print(f"{'Metric':<20} | {'Prompt Effect':<25} | {'Seed Effect':<25}")
        print(f"{'':<20} | {'F':>8} {'p':>8} {'η²p':>6} | {'F':>8} {'p':>8} {'η²p':>6}")
    print("-" * 90)
    
    for results in all_results:
        metric = results["metric"]
        display = METRIC_DISPLAY_NAMES.get(metric, metric)[:18]
        
        # Significance markers
        sig_p = "***" if results["prompt_p"] < 0.001 else "**" if results["prompt_p"] < 0.01 else "*" if results["prompt_p"] < 0.05 else ""
        sig_s = "***" if results["seed_p"] < 0.001 else "**" if results["seed_p"] < 0.01 else "*" if results["seed_p"] < 0.05 else ""
        
        if has_interaction and results["interaction_F"] is not None:
            sig_i = "***" if results["interaction_p"] < 0.001 else "**" if results["interaction_p"] < 0.01 else "*" if results["interaction_p"] < 0.05 else ""
            print(f"{display:<20} | "
                  f"{results['prompt_F']:>8.2f} {results['prompt_p']:>7.4f}{sig_p:<3} {results['prompt_eta_sq']:>.3f} | "
                  f"{results['seed_F']:>8.2f} {results['seed_p']:>7.4f}{sig_s:<3} {results['seed_eta_sq']:>.3f} | "
                  f"{results['interaction_F']:>8.2f} {results['interaction_p']:>6.4f}{sig_i}")
        else:
            print(f"{display:<20} | "
                  f"{results['prompt_F']:>8.2f} {results['prompt_p']:>7.4f}{sig_p:<3} {results['prompt_eta_sq']:>.3f} | "
                  f"{results['seed_F']:>8.2f} {results['seed_p']:>7.4f}{sig_s:<3} {results['seed_eta_sq']:>.3f}")
    
    print("-" * 90)
    print("Significance: * p < 0.05, ** p < 0.01, *** p < 0.001")
    print("η²p = partial eta-squared (effect size)")
    if not has_interaction:
        print("Note: Interaction not estimated (no replication); used as error term for F-tests")


def main():
    print("=" * 70)
    print("Two-Way ANOVA: Prompt Sensitivity Analysis")
    print("=" * 70)
    print(f"WandB Project: {WANDB_PROJECT}")
    print(f"LLM Model: {LLM_MODEL}")
    print(f"Prompt Variants: {len(PROMPT_VARIANTS)}")
    print(f"Seeds: {NUM_SEEDS}")
    print(f"Metrics: {METRICS}")
    print("=" * 70)
    
    # Fetch runs from wandb
    runs = fetch_runs()
    if not runs:
        print("No runs found! Exiting.")
        return
    
    # Build DataFrame
    print("\nBuilding data matrix...")
    df = build_dataframe(runs)
    
    if df.empty:
        print("No matching data found! Exiting.")
        return
    
    print(f"  Total observations: {len(df)}")
    print(f"  Models: {df['model'].unique().tolist()}")
    print(f"  Prompts: {df['prompt'].unique().tolist()}")
    print(f"  Seeds: {sorted(df['seed'].unique().tolist())}")
    
    # Run analysis for each model type
    for model_type in df["model"].unique():
        print(f"\n\n{'#' * 70}")
        print(f"# MODEL: {model_type.upper()}")
        print(f"{'#' * 70}")
        
        model_df = df[df["model"] == model_type].copy()
        
        # Check we have data for all combinations
        n_prompts = model_df["prompt"].nunique()
        n_seeds = model_df["seed"].nunique()
        print(f"\nData: {n_prompts} prompts × {n_seeds} seeds = {len(model_df)} observations")
        
        if n_prompts < 2:
            print(f"  Skipping: Need at least 2 prompt variants for ANOVA")
            continue
        
        # Run ANOVA for each metric
        all_results = []
        
        for metric in METRICS:
            # Check if metric exists in data
            if metric not in model_df.columns or model_df[metric].isna().all():
                print(f"\n  Skipping {metric}: No data available")
                continue
            
            # Drop rows with missing values for this metric
            metric_df = model_df[["prompt", "seed", metric]].dropna()
            
            if len(metric_df) < 10:
                print(f"\n  Skipping {metric}: Insufficient data ({len(metric_df)} observations)")
                continue
            
            # Print descriptive statistics
            print_descriptive_stats(metric_df, metric)
            
            # Run ANOVA
            try:
                results = run_two_way_anova(metric_df, metric)
                all_results.append(results)
                print_anova_results(results)
            except Exception as e:
                print(f"\n  Error running ANOVA for {metric}: {e}")
        
        # Print summary table
        if all_results:
            print_summary_table(all_results)
    
    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()

