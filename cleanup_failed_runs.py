"""Identify failed wandb runs that are safe to delete.

A run is considered safe to delete when ALL of the following hold:
  1. state is one of {failed, killed, crashed} (not 'finished' or 'running').
  2. It has no aggregate metrics (no avg/cv_wiki, avg/purity, etc.).
  3. There exists a 'finished' run with the same display name in the same
     project — i.e., the failed run has been superseded.

Run with --dry-run (default) to print candidates only.
Run with --execute to actually delete them.
"""
import argparse
import os
import sys
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv('/home/toolkit/LLM-Topic-Modeling/.env')
import wandb

ENTITY = os.environ['WANDB_ENTITY']
DATASETS = ['20_newsgroups', 'tweet_topic', 'stackoverflow', 'tweet_topic_detach_test']

NON_FINISHED_STATES = {'failed', 'killed', 'crashed'}
METRIC_KEYS = ['avg/cv_wiki', 'avg/cv', 'avg/purity', 'avg/llm_rating',
               'avg/inverted_rbo', 'avg/topic_diversity']


def has_any_metric(run):
    """Return True if the run has any aggregate metric in its summary."""
    for k in METRIC_KEYS:
        if run.summary.get(k) is not None:
            return True
    # Also check seed-level metrics
    for k in run.summary.keys():
        if k.startswith('seed_') and ('/purity' in k or '/cv' in k or '/llm_rating' in k):
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true',
                        help='Actually delete (default: dry-run only).')
    args = parser.parse_args()

    api = wandb.Api(timeout=60)

    # Group runs by (project, name) to find supersession relationships
    to_delete = []      # (project, run, reason)
    to_keep_failed = [] # (project, run, reason) — failed but kept (no successor or has data)

    for dataset in DATASETS:
        project = f"{ENTITY}/{dataset}"
        try:
            runs = list(api.runs(project, per_page=500))
        except Exception as e:
            print(f"Skipping {project}: {e}", file=sys.stderr)
            continue

        # Index runs by name
        by_name = defaultdict(list)
        for r in runs:
            by_name[r.name].append(r)

        # Track stats per project
        finished = sum(1 for r in runs if r.state == 'finished')
        non_finished = sum(1 for r in runs if r.state in NON_FINISHED_STATES)
        running = sum(1 for r in runs if r.state == 'running')
        print(f"\n[{dataset}] runs={len(runs)} finished={finished} "
              f"non_finished={non_finished} running={running}", file=sys.stderr)

        for name, group in by_name.items():
            # Skip group if any are running
            if any(r.state == 'running' for r in group):
                continue

            # Find a finished member of the group
            finished_runs = [r for r in group if r.state == 'finished']

            for r in group:
                if r.state == 'finished':
                    continue
                if r.state == 'running':
                    continue
                if r.state not in NON_FINISHED_STATES:
                    # unknown state, be conservative
                    to_keep_failed.append((dataset, r, f"unknown state '{r.state}'"))
                    continue
                if has_any_metric(r):
                    to_keep_failed.append((dataset, r, "has metrics — keep for safety"))
                    continue
                if not finished_runs:
                    to_keep_failed.append((dataset, r, "no finished sibling — would be deleting unique evidence"))
                    continue
                # All conditions satisfied
                to_delete.append((dataset, r, f"superseded by {finished_runs[0].id}"))

    # Print summary
    print(f"\n{'='*80}")
    print(f"FAILED-RUN CLEANUP {'(DRY RUN)' if not args.execute else '(EXECUTING)'}")
    print(f"{'='*80}")
    print(f"\nCandidates for deletion:  {len(to_delete)}")
    print(f"Failed runs kept (safety): {len(to_keep_failed)}")

    if to_delete:
        print(f"\n{'-'*80}")
        print(f"WILL DELETE ({len(to_delete)} runs):")
        print(f"{'-'*80}")
        by_proj = defaultdict(list)
        for proj, r, reason in to_delete:
            by_proj[proj].append((r, reason))
        for proj in sorted(by_proj.keys()):
            print(f"\n[{proj}]")
            for r, reason in sorted(by_proj[proj], key=lambda t: t[0].name):
                print(f"  id={r.id:10s} state={r.state:8s} {r.name:55s} ({reason})")

    if to_keep_failed:
        print(f"\n{'-'*80}")
        print(f"KEEPING (failed but cannot safely delete):")
        print(f"{'-'*80}")
        for proj, r, reason in to_keep_failed:
            print(f"  [{proj}] id={r.id:10s} state={r.state:8s} {r.name:55s} ({reason})")

    if args.execute and to_delete:
        print(f"\n{'='*80}")
        print(f"DELETING {len(to_delete)} runs...")
        print(f"{'='*80}")
        deleted = 0
        for proj, r, reason in to_delete:
            try:
                r.delete(delete_artifacts=True)
                deleted += 1
                print(f"  deleted {r.id} ({proj}/{r.name})")
            except Exception as e:
                print(f"  FAILED to delete {r.id}: {e}")
        print(f"\nDeleted {deleted}/{len(to_delete)} runs.")
    elif not args.execute and to_delete:
        print(f"\n--- DRY RUN: re-run with --execute to actually delete ---")


if __name__ == '__main__':
    main()
