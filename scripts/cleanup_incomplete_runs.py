#!/usr/bin/env python3
"""Cleanup script to identify and delete duplicate/incomplete runs from W&B.

Targets:
- 20_newsgroups: Duplicate generative_ERNIE-4.5-0.3B-PT K25 _CE (nll ablation) run
- tweet_topic: Incomplete Llama-3.2-1B-Instruct_nll K25, K50 runs (if any exist)
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
from settings import settings


def find_duplicate_runs(api, project: str, run_name_pattern: str, num_topics: int):
    """Find duplicate runs matching a pattern and K value."""
    runs = api.runs(
        f"{settings.wandb_entity}/{project}",
        filters={"state": "finished"},
    )
    
    matching_runs = []
    for run in runs:
        if run_name_pattern in run.name and run.config.get("num_topics") == num_topics:
            matching_runs.append(run)
    
    return matching_runs


def find_incomplete_nll_runs(api, project: str, llm_model: str):
    """Find incomplete NLL ablation runs for a specific LLM model.
    
    These are runs that exist but may have issues (e.g., missing seeds).
    """
    runs = api.runs(
        f"{settings.wandb_entity}/{project}",
        filters={"state": "finished"},
    )
    
    incomplete_runs = []
    for run in runs:
        # Match NLL ablation pattern: ends with _CE but no bow-target or embedding suffix
        if (f"generative_{llm_model}_K" in run.name and 
            run.name.endswith("_CE") and 
            "_bow-target" not in run.name and
            "_gte-large" not in run.name):
            
            # Check if it has the required number of seeds
            num_seeds = run.config.get("num_seeds", 0)
            if num_seeds != 5:
                incomplete_runs.append(run)
    
    return incomplete_runs


def main():
    parser = argparse.ArgumentParser(description="Cleanup duplicate/incomplete W&B runs")
    parser.add_argument("--dry-run", action="store_true", 
                        help="Preview deletions without actually deleting")
    parser.add_argument("--delete", action="store_true",
                        help="Actually delete the runs (requires confirmation)")
    args = parser.parse_args()
    
    if not args.dry_run and not args.delete:
        print("Please specify either --dry-run or --delete")
        sys.exit(1)
    
    api = wandb.Api()
    
    runs_to_delete = []
    
    # 1. Find duplicate ERNIE NLL ablation runs on 20_newsgroups K25
    print("\n=== Checking 20_newsgroups for duplicate ERNIE NLL K25 runs ===")
    ernie_nll_k25_runs = find_duplicate_runs(
        api, 
        "20_newsgroups", 
        "generative_ERNIE-4.5-0.3B-PT_K25_CE",  # NLL ablation pattern
        num_topics=25
    )
    
    if len(ernie_nll_k25_runs) > 1:
        print(f"Found {len(ernie_nll_k25_runs)} duplicate runs:")
        # Sort by creation time, keep the newest one
        ernie_nll_k25_runs.sort(key=lambda r: r.created_at, reverse=True)
        for i, run in enumerate(ernie_nll_k25_runs):
            status = "KEEP (newest)" if i == 0 else "DELETE"
            print(f"  [{status}] {run.name} (id: {run.id}, created: {run.created_at})")
            if i > 0:
                runs_to_delete.append(run)
    else:
        print(f"Found {len(ernie_nll_k25_runs)} runs (no duplicates)")
    
    # 2. Find incomplete Llama-1B NLL runs on tweet_topic
    print("\n=== Checking tweet_topic for incomplete Llama-1B NLL runs ===")
    incomplete_llama_runs = find_incomplete_nll_runs(
        api,
        "tweet_topic",
        "Llama-3.2-1B-Instruct"
    )
    
    if incomplete_llama_runs:
        print(f"Found {len(incomplete_llama_runs)} incomplete runs:")
        for run in incomplete_llama_runs:
            num_seeds = run.config.get("num_seeds", 0)
            num_topics = run.config.get("num_topics", "?")
            print(f"  [DELETE] {run.name} (id: {run.id}, seeds: {num_seeds}, K: {num_topics})")
            runs_to_delete.append(run)
    else:
        print("No incomplete runs found")
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Total runs to delete: {len(runs_to_delete)}")
    
    if not runs_to_delete:
        print("Nothing to delete.")
        return
    
    if args.dry_run:
        print("\n[DRY RUN] No runs were deleted. Use --delete to actually delete.")
        return
    
    # Confirm deletion
    if args.delete:
        print("\nRuns to be deleted:")
        for run in runs_to_delete:
            print(f"  - {run.project}/{run.name} (id: {run.id})")
        
        confirm = input("\nAre you sure you want to delete these runs? (yes/no): ")
        if confirm.lower() != "yes":
            print("Aborted.")
            return
        
        # Delete runs
        print("\nDeleting runs...")
        for run in runs_to_delete:
            try:
                run.delete()
                print(f"  Deleted: {run.name}")
            except Exception as e:
                print(f"  Error deleting {run.name}: {e}")
        
        print("\nDone!")


if __name__ == "__main__":
    main()

