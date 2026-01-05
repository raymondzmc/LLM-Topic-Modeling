#!/usr/bin/env python3
"""
Force delete ERNIE runs using the API client directly.
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
    
    total_deleted = 0
    
    for project_name in PROJECTS:
        full_project = f"{WANDB_ENTITY}/{project_name}"
        print(f"\n{'='*60}")
        print(f"Searching project: {full_project}")
        print(f"{'='*60}")
        
        # Fetch all runs from this project
        runs = api.runs(full_project)
        
        deleted_count = 0
        runs_to_delete = []
        
        # First collect all matching runs
        for run in runs:
            run_name = run.name
            if pattern.match(run_name):
                runs_to_delete.append((run, run_name, run.id))
        
        print(f"Found {len(runs_to_delete)} runs to delete")
        
        # Delete them
        for run, run_name, run_id in runs_to_delete:
            print(f"  Deleting: {run_name} (ID: {run_id})")
            try:
                # Try using the API client to delete
                api.client.execute(
                    """
                    mutation DeleteRun($id: ID!) {
                        deleteRun(input: {id: $id}) {
                            success
                        }
                    }
                    """,
                    variable_values={"id": run_id}
                )
                print(f"    DELETED ✓")
                deleted_count += 1
                total_deleted += 1
            except Exception as e:
                print(f"    ERROR: {e}")
                # Try the old method as fallback
                try:
                    run.delete()
                    print(f"    DELETED (fallback) ✓")
                    deleted_count += 1
                    total_deleted += 1
                except Exception as e2:
                    print(f"    FAILED: {e2}")
        
        print(f"\n  Deleted {deleted_count}/{len(runs_to_delete)} runs from {project_name}")
    
    print(f"\n{'='*60}")
    print(f"Total runs deleted: {total_deleted}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

