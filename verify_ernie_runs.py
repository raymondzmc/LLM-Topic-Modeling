#!/usr/bin/env python3
"""
Verify if ERNIE runs with exact pattern still exist in W&B.
"""

import re
import os
import wandb

# Get wandb entity from environment
WANDB_ENTITY = os.getenv('WANDB_ENTITY', 'llm-topics')

# Projects to search
PROJECTS = ['20_newsgroups', 'tweet_topic', 'stackoverflow']

# Build regex pattern for exact match (no suffixes)
pattern = re.compile(r'^generative_ERNIE-4\.5-0\.3B-PT_K(25|50|75|100)$')

def main():
    api = wandb.Api()
    
    total_found = 0
    
    for project_name in PROJECTS:
        full_project = f"{WANDB_ENTITY}/{project_name}"
        print(f"\nSearching project: {full_project}")
        
        # Fetch all runs from this project
        runs = api.runs(full_project)
        
        found_count = 0
        for run in runs:
            run_name = run.name
            
            # Check if run name matches the exact pattern (no suffixes)
            if pattern.match(run_name):
                print(f"  STILL EXISTS: {run_name} (ID: {run.id}, State: {run.state})")
                found_count += 1
                total_found += 1
        
        if found_count == 0:
            print(f"  ✓ No matching runs found in {project_name}")
        else:
            print(f"  Found {found_count} runs in {project_name}")
    
    print(f"\n{'='*60}")
    if total_found == 0:
        print("✓ All runs successfully deleted!")
    else:
        print(f"⚠️  WARNING: {total_found} runs still exist")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

