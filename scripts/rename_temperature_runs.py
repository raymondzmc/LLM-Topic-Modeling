#!/usr/bin/env python
"""
Rename W&B runs to include temperature suffix for temperature ablation experiments.

This script fetches all generative model runs, checks if they have a non-default
temperature (not 3.0), and renames them to include '_temp{temperature}' suffix.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import wandb
from settings import settings


def rename_temperature_runs(
    projects: list[str],
    dry_run: bool = True,
    model_filter: str | None = None,
):
    """
    Rename runs with non-default temperature to include temperature suffix.
    
    Args:
        projects: List of W&B project names to process
        dry_run: If True, only print what would be renamed without making changes
        model_filter: Optional model name filter (e.g., 'ERNIE-4.5-0.3B-PT')
    """
    api = wandb.Api()
    
    total_renamed = 0
    total_skipped = 0
    
    for project in projects:
        project_path = f"{settings.wandb_entity}/{project}"
        print(f"\n{'='*60}")
        print(f"Processing project: {project_path}")
        print(f"{'='*60}")
        
        try:
            runs = api.runs(project_path)
        except wandb.errors.CommError as e:
            print(f"  Error fetching runs: {e}")
            continue
        
        runs_list = list(runs)
        print(f"  Found {len(runs_list)} total runs")
        
        renamed_count = 0
        skipped_count = 0
        
        for run in runs_list:
            config = run.config
            current_name = run.name
            
            # Check if this is a generative model run
            model = config.get('model', '')
            if model != 'generative':
                continue
            
            # Apply model filter if specified
            if model_filter:
                # Check if model name is in the run name or config
                dataset_model = current_name.split('_K')[0] if '_K' in current_name else ''
                if model_filter not in dataset_model and model_filter not in current_name:
                    continue
            
            # Get temperature from config
            temperature = config.get('temperature', 3.0)
            
            # Skip if temperature is default (3.0)
            if temperature == 3.0:
                continue
            
            # Check if already has temperature suffix
            temp_suffix = f"_temp{temperature}"
            if temp_suffix in current_name:
                skipped_count += 1
                continue
            
            # Build new name with temperature suffix
            # Insert before any existing suffix patterns or at end
            new_name = current_name + temp_suffix
            
            print(f"  [{run.id}] {current_name} -> {new_name}")
            
            if not dry_run:
                run.name = new_name
                run.update()
                print(f"    ✓ Renamed")
            else:
                print(f"    (dry run - no changes made)")
            
            renamed_count += 1
        
        total_renamed += renamed_count
        total_skipped += skipped_count
        
        print(f"\n  Summary for {project}:")
        print(f"    - Runs to rename: {renamed_count}")
        print(f"    - Already have suffix: {skipped_count}")
    
    print(f"\n{'='*60}")
    print(f"TOTAL SUMMARY")
    print(f"{'='*60}")
    print(f"  - Total runs renamed: {total_renamed}")
    print(f"  - Total runs skipped (already have suffix): {total_skipped}")
    
    if dry_run and total_renamed > 0:
        print(f"\n  ⚠️  This was a dry run. Use --execute to apply changes.")


def main():
    parser = argparse.ArgumentParser(
        description="Rename W&B runs to include temperature suffix"
    )
    parser.add_argument(
        '--projects',
        nargs='+',
        default=['20_newsgroups', 'tweet_topic', 'stackoverflow'],
        help='W&B project names to process'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Filter by model name (e.g., ERNIE-4.5-0.3B-PT)'
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually rename runs (default is dry run)'
    )
    
    args = parser.parse_args()
    
    print("W&B Run Renamer - Temperature Suffix")
    print(f"Entity: {settings.wandb_entity}")
    print(f"Projects: {args.projects}")
    print(f"Model filter: {args.model or 'None (all generative runs)'}")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN'}")
    
    rename_temperature_runs(
        projects=args.projects,
        dry_run=not args.execute,
        model_filter=args.model,
    )


if __name__ == '__main__':
    main()

