"""Aggregate metrics for the DSL paper table: 3 backbones × 3 LMs × 3 datasets.

For each (backbone, LM, dataset), average over 4 K values × 5 seeds = 20 runs.
Verify completeness and report missing seeds/runs.
"""
import os
import sys
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv('/home/toolkit/LLM-Topic-Modeling/.env')
import wandb

ENTITY = os.environ['WANDB_ENTITY']
DATASETS_WB = ['20_newsgroups', 'tweet_topic', 'stackoverflow']

BACKBONES = {
    'ProdLDA + DSL':   'generative',
    'FASTopic + DSL':  'generative_fastopic',
    'ECRTM + DSL':     'generative_ecrtm',
}
LMS_ORDER = ['ERNIE-4.5-0.3B-PT', 'Qwen3.5-0.8B', 'Llama-3.2-1B-Instruct']
LM_DISPLAY = {
    'ERNIE-4.5-0.3B-PT':       'ERNIE-4.5-0.3B',
    'Qwen3.5-0.8B':            'Qwen3.5-0.8B',
    'Llama-3.2-1B-Instruct':   'Llama-3.2-1B',
}
KS = [25, 50, 75, 100]
N_SEEDS = 5

METRICS = {
    'cv':     'avg/cv_wiki',
    'llm':    'avg/llm_rating',
    'irbo':   'avg/inverted_rbo',
    'purity': 'avg/purity',
}


def count_seeds(summary):
    """Count distinct seed indices in per-seed metric keys."""
    seen = set()
    for key in summary.keys():
        if key.startswith('seed_') and '/' in key:
            try:
                seen.add(int(key.split('/', 1)[0].split('_', 1)[1]))
            except (ValueError, IndexError):
                pass
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
    return (non_degenerate, finished, has_cv, len(count_seeds(s)), ari)


api = wandb.Api(timeout=60)

# r[(dataset, backbone, lm, K)] = {metric: value, seeds: set}
r = {}
missing = []

for dataset in DATASETS_WB:
    project = f"{ENTITY}/{dataset}"
    print(f"Fetching {project}...", file=sys.stderr)
    runs = list(api.runs(project, per_page=500))
    by_name = {}
    for run in runs:
        ex = by_name.get(run.name)
        if ex is None or quality(run) > quality(ex):
            by_name[run.name] = run

    for label, backbone in BACKBONES.items():
        for lm in LMS_ORDER:
            for K in KS:
                name = f"{backbone}_{lm}_K{K}"
                run = by_name.get(name)
                if run is None or run.state != 'finished':
                    missing.append((dataset, label, lm, K,
                                    'no run' if run is None else f'state={run.state}'))
                    continue
                vals = {}
                for short, key in METRICS.items():
                    v = run.summary.get(key)
                    if v is not None:
                        vals[short] = float(v)
                seeds = count_seeds(run.summary)
                r[(dataset, label, lm, K)] = {'vals': vals, 'seeds': seeds, 'run_id': run.id}


# Completeness report
print(f"\n{'='*70}")
print(f"COMPLETENESS CHECK")
print(f"{'='*70}")
expected = len(DATASETS_WB) * len(BACKBONES) * len(LMS_ORDER) * len(KS)
print(f"Expected: {expected} runs ({len(DATASETS_WB)} datasets × {len(BACKBONES)} backbones × {len(LMS_ORDER)} LMs × {len(KS)} K)")
print(f"Found:    {len(r)} runs")
print(f"Missing:  {len(missing)}")

if missing:
    print(f"\n--- Missing runs ---")
    for d, lab, lm, K, why in missing:
        print(f"  {d:20s} {lab:18s} {lm:25s} K={K:3d}  ({why})")

# Check seed counts
seed_issues = []
metric_issues = []
for key, info in r.items():
    if len(info['seeds']) != N_SEEDS:
        seed_issues.append((key, len(info['seeds']), sorted(info['seeds'])))
    for m in METRICS:
        if m not in info['vals']:
            metric_issues.append((key, m))

if seed_issues:
    print(f"\n--- Runs with <{N_SEEDS} seeds ---")
    for key, n, s in seed_issues:
        d, lab, lm, K = key
        print(f"  {d:20s} {lab:18s} {lm:25s} K={K:3d}  seeds={s}")
else:
    print(f"\nAll {len(r)} runs have {N_SEEDS} seeds.")

if metric_issues:
    print(f"\n--- Runs with missing metrics ---")
    for key, m in metric_issues:
        d, lab, lm, K = key
        print(f"  {d:20s} {lab:18s} {lm:25s} K={K:3d}  missing={m}")
else:
    print(f"All metrics present.")


# Aggregate across K values
def avg_over_k(dataset, label, lm, metric):
    vals = []
    for K in KS:
        info = r.get((dataset, label, lm, K))
        if info and metric in info['vals']:
            vals.append(info['vals'][metric])
    return sum(vals) / len(vals) if vals else None, len(vals)


# Formatting helpers
def fmt_cv_pur(v):
    if v is None: return '—'
    return f".{int(round(v*1000)):03d}"

def fmt_llm(v):
    if v is None: return '—'
    return f"{v:.2f}"

def fmt_irbo(v):
    if v is None: return '—'
    if v >= 0.9995: return "1.000"
    return f".{int(round(v*1000)):03d}"


# Display: per-cell averages
print(f"\n{'='*70}")
print(f"AGGREGATED METRICS (averaged over K=25,50,75,100 × 5 seeds)")
print(f"{'='*70}")

DATASETS_DISPLAY = ['20_newsgroups', 'tweet_topic', 'stackoverflow']

for label in BACKBONES.keys():
    print(f"\n## {label}")
    print(f"{'LM':18s} | " + " | ".join(f"{d:38s}" for d in DATASETS_DISPLAY))
    print(f"{'':18s} | " + " | ".join("CV     LLM    I-RBO  Purity         " for _ in DATASETS_DISPLAY))
    for lm in LMS_ORDER:
        cells = []
        for d in DATASETS_DISPLAY:
            cv, n1 = avg_over_k(d, label, lm, 'cv')
            ll, n2 = avg_over_k(d, label, lm, 'llm')
            ir, n3 = avg_over_k(d, label, lm, 'irbo')
            pu, n4 = avg_over_k(d, label, lm, 'purity')
            cells.append(f"{fmt_cv_pur(cv):6s} {fmt_llm(ll):6s} {fmt_irbo(ir):6s} {fmt_cv_pur(pu):6s} (n={n1})")
        print(f"{LM_DISPLAY[lm]:18s} | " + " | ".join(cells))


# Also emit LaTeX rows
print(f"\n{'='*70}")
print(f"LATEX ROWS  (paste under 'Ours' section)")
print(f"{'='*70}")

for label in BACKBONES.keys():
    for lm in LMS_ORDER:
        cells = []
        for d in DATASETS_DISPLAY:
            cv, _ = avg_over_k(d, label, lm, 'cv')
            ll, _ = avg_over_k(d, label, lm, 'llm')
            ir, _ = avg_over_k(d, label, lm, 'irbo')
            pu, _ = avg_over_k(d, label, lm, 'purity')
            cells.append(f"{fmt_cv_pur(cv)} & {fmt_llm(ll)} & {fmt_irbo(ir)} & {fmt_cv_pur(pu)}")
        row = f"\\texttt{{{LM_DISPLAY[lm]}}} & " + " & ".join(cells) + r" \\"
        print(row)
    print(r"\midrule  % " + label)
