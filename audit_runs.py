"""Audit wandb runs: find missing runs and runs with incomplete seeds.

Expected full matrix:
  * Baselines: {lda, prodlda, zeroshot, combined, etm, bertopic, ecrtm, fastopic}
    × 3 datasets × 4 K values (no LLM suffix)
  * Generative (3 backbones) × 5 LLMs × 3 datasets × 4 K values
Each run is expected to have 5 seeds (seed_0..seed_4).
"""
import os
import sys
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv('/home/toolkit/LLM-Topic-Modeling/.env')
import wandb

ENTITY = os.environ['WANDB_ENTITY']
DATASETS = ['20_newsgroups', 'tweet_topic', 'stackoverflow']
BACKBONE_MODELS = ['generative', 'generative_ecrtm', 'generative_fastopic']
LLMS = [
    'ERNIE-4.5-0.3B-PT',
    'Llama-3.1-8B-Instruct',
    'Llama-3.2-1B-Instruct',
    'Phi-3-mini-128k-instruct',
    'Qwen3.5-0.8B',
]
BASELINES = ['lda', 'prodlda', 'zeroshot', 'combined', 'etm', 'bertopic', 'ecrtm', 'fastopic']
KS = [25, 50, 75, 100]
EXPECTED_SEEDS = 5

# Seed presence detector: any per-seed key counts
def count_seeds(summary):
    seen = set()
    for key in summary.keys():
        if not key.startswith('seed_'):
            continue
        # format: seed_<N>/<metric>
        try:
            idx = int(key.split('/', 1)[0].split('_', 1)[1])
            seen.add(idx)
        except (ValueError, IndexError):
            continue
    return seen

def quality(run):
    s = run.summary
    finished = 1 if run.state == 'finished' else 0
    has_cv = 1 if s.get('avg/cv_wiki') is not None else 0
    ari = s.get('avg/ari') or 0
    inv_purity = s.get('avg/inverse_purity')
    non_degenerate = 1 if (ari and ari > 0) else 0
    if inv_purity is not None and inv_purity >= 0.999 and (ari is None or ari == 0):
        non_degenerate = 0
    seeds = len(count_seeds(s))
    return (non_degenerate, finished, has_cv, seeds)

api = wandb.Api(timeout=60)

# Build expected run name set
expected = {}  # (dataset, run_name) -> category
for dataset in DATASETS:
    # Baselines
    for m in BASELINES:
        for K in KS:
            expected[(dataset, f"{m}_K{K}")] = ('baseline', m, None, K)
    # Generative (backbone × LLM)
    for m in BACKBONE_MODELS:
        for llm in LLMS:
            for K in KS:
                expected[(dataset, f"{m}_{llm}_K{K}")] = ('generative', m, llm, K)

# Also check if each expected wandb project exists
print(f"Entity: {ENTITY}", file=sys.stderr)
missing_projects = []
runs_by_project = {}
for dataset in DATASETS:
    project = f"{ENTITY}/{dataset}"
    try:
        runs = list(api.runs(project, per_page=500))
        runs_by_project[dataset] = runs
        print(f"  {dataset}: {len(runs)} runs", file=sys.stderr)
    except Exception as e:
        missing_projects.append((dataset, str(e)))
        print(f"  {dataset}: ERROR {e}", file=sys.stderr)

# Dedup runs by name within each project, keeping highest-quality
best = {}  # (dataset, name) -> run
for dataset, runs in runs_by_project.items():
    for run in runs:
        # Skip ablation/sweep runs (names contain suffixes like _temp, _topk, _bow-target,
        # _sparsity, _CE, or include a gte-large embedding suffix)
        name = run.name
        # Keep only exact-matching expected names
        if (dataset, name) not in expected:
            continue
        ex = best.get((dataset, name))
        if ex is None or quality(run) > quality(ex):
            best[(dataset, name)] = run

# Build report buckets
missing_runs = []       # no run at all (or all failed)
failed_runs = []        # latest matching run is failed
incomplete_seeds = []   # run exists but seed count < EXPECTED_SEEDS
degenerate_runs = []    # run with avg/ari == 0 and inverse_purity == 1
complete_runs = []

for key, category in expected.items():
    dataset, name = key
    run = best.get(key)
    if run is None:
        # search among ALL runs (not just deduped) for any attempt
        any_attempt = [r for r in runs_by_project.get(dataset, []) if r.name == name]
        if any_attempt:
            # All attempts failed
            failed_runs.append((dataset, name, category, [r.state for r in any_attempt]))
        else:
            missing_runs.append((dataset, name, category))
        continue

    s = run.summary
    seeds = count_seeds(s)
    ari = s.get('avg/ari') or 0
    inv_purity = s.get('avg/inverse_purity')
    degenerate = inv_purity is not None and inv_purity >= 0.999 and (ari is None or ari == 0)

    if run.state != 'finished':
        failed_runs.append((dataset, name, category, [run.state]))
    elif degenerate:
        degenerate_runs.append((dataset, name, category, run.state, sorted(seeds)))
    elif len(seeds) < EXPECTED_SEEDS:
        incomplete_seeds.append((dataset, name, category, run.state, sorted(seeds)))
    else:
        complete_runs.append((dataset, name, category, run.state, sorted(seeds)))

# Print report
print("\n" + "="*80)
print("AUDIT REPORT")
print("="*80)

print(f"\nDatasets checked: {DATASETS}")
if missing_projects:
    print(f"\nMISSING wandb projects: {missing_projects}")
else:
    print(f"All {len(DATASETS)} wandb projects accessible.")

total_expected = len(expected)
print(f"\nExpected runs: {total_expected}")
print(f"  Complete (5/5 seeds, finished): {len(complete_runs)}")
print(f"  Failed / non-finished state:   {len(failed_runs)}")
print(f"  Incomplete seeds:              {len(incomplete_seeds)}")
print(f"  Degenerate (ARI=0, inv=1):     {len(degenerate_runs)}")
print(f"  No run at all:                 {len(missing_runs)}")

def fmt_cat(cat):
    kind, m, llm, K = cat
    if kind == 'baseline':
        return f"{m:10s} K={K}"
    else:
        return f"{m:20s} LLM={llm:30s} K={K}"

if missing_runs:
    print("\n" + "-"*80)
    print(f"MISSING RUNS ({len(missing_runs)}):")
    print("-"*80)
    # Group by category for readability
    by_cat = defaultdict(list)
    for dataset, name, category in missing_runs:
        kind, m, llm, K = category
        if kind == 'baseline':
            group = f"baseline:{m}"
        else:
            group = f"{m}/{llm}"
        by_cat[group].append((dataset, K))
    for group in sorted(by_cat.keys()):
        print(f"\n  [{group}]")
        for dataset, K in sorted(by_cat[group]):
            print(f"    {dataset:20s} K={K}")

if failed_runs:
    print("\n" + "-"*80)
    print(f"FAILED / NON-FINISHED RUNS ({len(failed_runs)}):")
    print("-"*80)
    by_cat = defaultdict(list)
    for dataset, name, category, states in failed_runs:
        kind, m, llm, K = category
        if kind == 'baseline':
            group = f"baseline:{m}"
        else:
            group = f"{m}/{llm}"
        by_cat[group].append((dataset, K, states))
    for group in sorted(by_cat.keys()):
        print(f"\n  [{group}]")
        for dataset, K, states in sorted(by_cat[group], key=lambda t: (t[0], t[1])):
            print(f"    {dataset:20s} K={K}  (states: {states})")

if incomplete_seeds:
    print("\n" + "-"*80)
    print(f"INCOMPLETE SEEDS ({len(incomplete_seeds)}):")
    print("-"*80)
    for dataset, name, category, state, seeds in sorted(incomplete_seeds):
        kind, m, llm, K = category
        present = ','.join(str(x) for x in seeds)
        missing_seeds = [i for i in range(EXPECTED_SEEDS) if i not in set(seeds)]
        print(f"  {dataset:20s} {name:55s} state={state:10s} seeds_present=[{present}] missing={missing_seeds}")

if degenerate_runs:
    print("\n" + "-"*80)
    print(f"DEGENERATE RUNS (ARI=0, clustering collapsed) ({len(degenerate_runs)}):")
    print("-"*80)
    for dataset, name, category, state, seeds in sorted(degenerate_runs):
        kind, m, llm, K = category
        print(f"  {dataset:20s} {name:55s} seeds={len(seeds)}/5")

# Summary matrix view
print("\n" + "="*80)
print("COMPLETION MATRIX (✓=ok, ✗=missing, ⚠=failed, ∘=incomplete seeds, ⨯=degenerate)")
print("="*80)

def status(key):
    dataset, name = key
    if any(key == (d, n) for d, n, _, _, _ in complete_runs):
        return '✓'
    if any(key == (d, n) for d, n, _, _ in failed_runs):
        return '⚠'
    if any(key == (d, n) for d, n, _, _, _ in incomplete_seeds):
        return '∘'
    if any(key == (d, n) for d, n, _, _, _ in degenerate_runs):
        return '⨯'
    return '✗'

for kind_label, groups in [
    ('BASELINES', [(m, None) for m in BASELINES]),
    ('GENERATIVE', [(m, llm) for m in BACKBONE_MODELS for llm in LLMS]),
]:
    print(f"\n{kind_label}")
    header = f"{'model/llm':45s} | " + " | ".join(f"{d:15s}" for d in DATASETS)
    print(header)
    print('-' * len(header))
    for m, llm in groups:
        if llm is None:
            row_label = f"{m}"
        else:
            row_label = f"{m} / {llm}"
        cells = []
        for dataset in DATASETS:
            per_K = []
            for K in KS:
                name = f"{m}_K{K}" if llm is None else f"{m}_{llm}_K{K}"
                per_K.append(status((dataset, name)))
            cells.append(' '.join(per_K))
        print(f"{row_label:45s} | " + " | ".join(f"{c:15s}" for c in cells))

print(f"\n(columns = K=25 K=50 K=75 K=100 within each dataset)")
