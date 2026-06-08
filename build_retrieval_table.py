"""Build the retrieval results table (table2.md / table2.tex).

Mirrors table.md's row structure: 8 baselines + 15 "Ours" rows
(5 LMs × {ProdLDA implicit, + ECRTM, + FASTopic}). Aggregated over
K=25/50/75/100 × 5 seeds = 20 per-cell measurements.
Welch's t-test for blue highlighting at alpha=0.05.
"""
import os
import sys
from collections import defaultdict
from dotenv import load_dotenv
import numpy as np
from scipy import stats

load_dotenv('/home/toolkit/LLM-Topic-Modeling/.env')
import wandb

ENTITY = os.environ['WANDB_ENTITY']
DATASETS = [('20_newsgroups', '20News'),
            ('tweet_topic',   'Tweet'),
            ('stackoverflow', 'Stack')]
KS = [25, 50, 75, 100]
N_SEEDS = 5
ALPHA = 0.05

# (display_label, kind, model_id, lm_id)
# kind: 'baseline' -> "{model}_K{K}"
# kind: 'ours'     -> "{model}_{lm}_K{K}"
# kind: 'group'    -> midrule + group header
ROWS = [
    ('LDA',        'baseline', 'lda',      None),
    ('ProdLDA',    'baseline', 'prodlda',  None),
    ('ZeroShotTM', 'baseline', 'zeroshot', None),
    ('CombinedTM', 'baseline', 'combined', None),
    ('ETM',        'baseline', 'etm',      None),
    ('BERTopic',   'baseline', 'bertopic', None),
    ('ECRTM',      'baseline', 'ecrtm',    None),
    ('FASTopic',   'baseline', 'fastopic', None),
    # Ours: ProdLDA + DSL (no suffix in label)
    ('OURS_HEADER', 'group', None, None),
    ('ERNIE-4.5-0.3B', 'ours', 'generative', 'ERNIE-4.5-0.3B-PT'),
    ('Llama-3.1-8B',   'ours', 'generative', 'Llama-3.1-8B-Instruct'),
    ('Llama-3.2-1B',   'ours', 'generative', 'Llama-3.2-1B-Instruct'),
    ('Qwen-3.5-0.8B',  'ours', 'generative', 'Qwen3.5-0.8B'),
    ('Phi-3-mini',     'ours', 'generative', 'Phi-3-mini-128k-instruct'),
    # Ours: ECRTM + DSL
    ('ERNIE-4.5-0.3B + ECRTM', 'ours', 'generative_ecrtm', 'ERNIE-4.5-0.3B-PT'),
    ('Llama-3.1-8B + ECRTM',   'ours', 'generative_ecrtm', 'Llama-3.1-8B-Instruct'),
    ('Llama-3.2-1B + ECRTM',   'ours', 'generative_ecrtm', 'Llama-3.2-1B-Instruct'),
    ('Qwen-3.5-0.8B + ECRTM',  'ours', 'generative_ecrtm', 'Qwen3.5-0.8B'),
    ('Phi-3-mini + ECRTM',     'ours', 'generative_ecrtm', 'Phi-3-mini-128k-instruct'),
    # Ours: FASTopic + DSL
    ('ERNIE-4.5-0.3B + FASTopic', 'ours', 'generative_fastopic', 'ERNIE-4.5-0.3B-PT'),
    ('Llama-3.1-8B + FASTopic',   'ours', 'generative_fastopic', 'Llama-3.1-8B-Instruct'),
    ('Llama-3.2-1B + FASTopic',   'ours', 'generative_fastopic', 'Llama-3.2-1B-Instruct'),
    ('Qwen-3.5-0.8B + FASTopic',  'ours', 'generative_fastopic', 'Qwen3.5-0.8B'),
    ('Phi-3-mini + FASTopic',     'ours', 'generative_fastopic', 'Phi-3-mini-128k-instruct'),
]

METRICS = [('p5',  'precision@5',  'P@5'),
           ('p10', 'precision@10', 'P@10')]


def run_name(kind, model, lm, K):
    if kind == 'baseline':
        return f"{model}_K{K}"
    if kind == 'ours':
        return f"{model}_{lm}_K{K}"
    return None


def quality(r):
    s = r.summary
    finished = 1 if r.state == 'finished' else 0
    has_cv = 1 if s.get('avg/cv_wiki') is not None else 0
    ari = s.get('avg/ari') or 0
    inv_purity = s.get('avg/inverse_purity')
    non_degenerate = 1 if (ari and ari > 0) else 0
    if inv_purity is not None and inv_purity >= 0.999 and (ari is None or ari == 0):
        non_degenerate = 0
    has_retrieval = 1 if s.get('retrieval/avg_precision@5') is not None else 0
    return (non_degenerate, finished, has_cv, has_retrieval, ari)


api = wandb.Api(timeout=60)
runs_by_name = {}
for ds_id, _ in DATASETS:
    print(f"Fetching {ds_id}...", file=sys.stderr)
    for r in api.runs(f"{ENTITY}/{ds_id}", per_page=500):
        key = (ds_id, r.name)
        ex = runs_by_name.get(key)
        if ex is None or quality(r) > quality(ex):
            runs_by_name[key] = r


# Collect per-seed-per-K retrieval values
data = defaultdict(list)
missing_retrieval = []
missing_runs = []
for row_idx, (label, kind, model, lm) in enumerate(ROWS):
    if kind == 'group':
        continue
    for ds_id, _ in DATASETS:
        for K in KS:
            name = run_name(kind, model, lm, K)
            run = runs_by_name.get((ds_id, name))
            if run is None or run.state != 'finished':
                missing_runs.append((ds_id, name, label))
                continue
            for short, full, _ in METRICS:
                if run.summary.get(f'retrieval/avg_{full}') is None:
                    missing_retrieval.append((ds_id, name, label, full))
                    continue
                for seed in range(N_SEEDS):
                    v = run.summary.get(f'retrieval/seed_{seed}/{full}')
                    if v is not None:
                        data[(row_idx, ds_id, short)].append(float(v))


print(f"\nCollected {len(data)} cells; expected {sum(1 for _, k, *_ in ROWS if k != 'group') * len(DATASETS) * len(METRICS)}")

# Sanity check: cells should have N_SEEDS * len(KS) = 20 values
short_cells = []
for key, vs in data.items():
    if len(vs) != N_SEEDS * len(KS):
        short_cells.append((key, len(vs)))
if short_cells:
    print(f"\nWARNING: {len(short_cells)} cells have fewer than {N_SEEDS*len(KS)} values:")
    for key, n in short_cells[:20]:
        ri, ds, m = key
        label = ROWS[ri][0]
        print(f"  {ds:18s} {label:30s} {m}: {n}/20")

if missing_retrieval:
    by = defaultdict(list)
    for ds, name, label, full in missing_retrieval:
        by[(label, full)].append(f"{ds}/{name}")
    print(f"\nMissing retrieval metrics ({len(missing_retrieval)} cell-level gaps):")
    for (label, full), names in sorted(by.items())[:20]:
        print(f"  {label:30s} {full:14s}: {len(names)} runs (e.g. {names[0]})")

if missing_runs:
    print(f"\nMissing runs (training itself absent): {len(missing_runs)}")
    for ds, name, label in missing_runs[:10]:
        print(f"  {ds}/{name}")


# Compute means + significance
means = {}
highlight = {}

for ds_id, _ in DATASETS:
    for short, _, _ in METRICS:
        per_row = {}
        for row_idx, (label, kind, *_) in enumerate(ROWS):
            if kind == 'group':
                continue
            vs = data.get((row_idx, ds_id, short))
            if not vs:
                continue
            per_row[row_idx] = np.array(vs)
        if not per_row:
            continue
        means_per_row = {ri: arr.mean() for ri, arr in per_row.items()}
        best_idx = max(means_per_row, key=means_per_row.get)
        best_vals = per_row[best_idx]
        for ri, arr in per_row.items():
            means[(ri, ds_id, short)] = means_per_row[ri]
            if ri == best_idx:
                highlight[(ri, ds_id, short)] = True
            else:
                t, p = stats.ttest_ind(arr, best_vals, equal_var=False)
                highlight[(ri, ds_id, short)] = bool(p >= ALPHA)


# Format helpers (P@k: ".XXX" form like table.md style)
def fmt(v):
    if v is None:
        return '---'
    return f".{int(round(v*1000)):03d}"


def cell(row_idx, ds_id, short):
    v = means.get((row_idx, ds_id, short))
    if v is None:
        return '---'
    txt = fmt(v)
    if highlight.get((row_idx, ds_id, short)):
        return rf'\colorbox{{lightblue}}{{\textbf{{{txt}}}}}'
    return txt


# Build LaTeX (match table2.md style: table[t], 6 metric columns)
out = []
out.append(r'\begin{table}[t]%[ht]')
out.append(r'\centering')
out.append(r'\begin{adjustbox}{width=\linewidth}')
out.append(r'\begin{tabular}{l *{6}{c}}')
out.append(r'\toprule')
out.append(r'\multicolumn{1}{c}{} &')
out.append(r'  \multicolumn{2}{c}{\large \textbf{20News}} &')
out.append(r'  \multicolumn{2}{c}{\large \textbf{Tweet}} &')
out.append(r'  \multicolumn{2}{c}{\large \textbf{Stack}} \\')
out.append(r'\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}')
out.append(r'\multicolumn{1}{c}{} &')
header_cells = []
for _ in DATASETS:
    header_cells.append(r'  \texttt{\small P@5}')
    header_cells.append(r'  \texttt{\small P@10}')
out.append(' &\n'.join(header_cells) + r' \\')
out.append(r'\midrule')

for row_idx, (label, kind, *_) in enumerate(ROWS):
    if kind == 'group':
        out.append(r'\midrule')
        out.append(r'\multicolumn{7}{c}{\large \textit{Ours}} \\')
        out.append(r'\midrule')
        continue

    cells = []
    for ds_id, _ in DATASETS:
        for short, _, _ in METRICS:
            cells.append(cell(row_idx, ds_id, short))

    out.append(label + ' &\n' + ' & '.join(cells) + r' \\')

out.append(r'\bottomrule')
out.append(r'\end{tabular}')
out.append(r'\end{adjustbox}')
out.append(r'\caption{')
out.append(r'Retrieval evaluation (P@5, P@10), averaged over four numbers of topics '
           r'($K=25, 50, 75, 100$) $\times$ five random seeds (20 measurements per cell).')
out.append(r"For each dataset and metric, methods are compared using Welch's independent samples "
           r"$t$-test ($\alpha=0.05$). The best result and those not statistically significantly "
           r'different from it ($p \ge 0.05$) are \colorbox{lightblue}{\textbf{highlighted in blue}}.')
out.append(r'Rows without a suffix use the ProdLDA backbone; rows suffixed with '
           r'\texttt{+ ECRTM} or \texttt{+ FASTopic} use the ECRTM or FASTopic backbone.')
out.append(r'}')
out.append(r'\label{tab:retrieval-results}')
out.append(r'\vspace{-1em}')
out.append(r'\end{table}')

latex = '\n'.join(out) + '\n'
for path in ['/home/toolkit/LLM-Topic-Modeling/table2.md',
             '/home/toolkit/LLM-Topic-Modeling/table2.tex']:
    with open(path, 'w') as f:
        f.write(latex)
    print(f"Wrote {path}")

# Print row-level highlight summary
print(f"\n--- Significance summary (highlights / 6 cells per row) ---")
for row_idx, (label, kind, *_) in enumerate(ROWS):
    if kind == 'group':
        print(f"  -- Ours --")
        continue
    hl = sum(1 for ds_id, _ in DATASETS for short, *_ in METRICS
             if highlight.get((row_idx, ds_id, short)))
    print(f"  {label:32s} {hl}/6")
