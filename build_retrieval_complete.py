"""Build the complete (per-K) retrieval results table — mirror of table.md.

Same 23 rows × 6 metric columns × 4 K sections (K=25/50/75/100), with
Welch's t-test highlighting computed *within each K* (best per dataset×metric
plus methods with p>=0.05 vs the best).

Outputs:
  /home/toolkit/LLM-Topic-Modeling/table_retrieval_complete.md
  /home/toolkit/LLM-Topic-Modeling/table_retrieval_complete.tex
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

# Row schema mirrors build_retrieval_table.py (matches table.md ordering)
ROWS = [
    ('LDA',        'baseline', 'lda',      None),
    ('ProdLDA',    'baseline', 'prodlda',  None),
    ('ZeroShotTM', 'baseline', 'zeroshot', None),
    ('CombinedTM', 'baseline', 'combined', None),
    ('ETM',        'baseline', 'etm',      None),
    ('BERTopic',   'baseline', 'bertopic', None),
    ('ECRTM',      'baseline', 'ecrtm',    None),
    ('FASTopic',   'baseline', 'fastopic', None),
    # Ours: ProdLDA + DSL (no suffix)
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
    ari = s.get('avg/ari') or 0
    inv_purity = s.get('avg/inverse_purity')
    non_degenerate = 1 if (ari and ari > 0) else 0
    if inv_purity is not None and inv_purity >= 0.999 and (ari is None or ari == 0):
        non_degenerate = 0
    has_retrieval = 1 if s.get('retrieval/avg_precision@5') is not None else 0
    return (non_degenerate, finished, has_retrieval, ari)


api = wandb.Api(timeout=60)
runs_by_name = {}
for ds_id, _ in DATASETS:
    print(f"Fetching {ds_id}...", file=sys.stderr)
    for r in api.runs(f"{ENTITY}/{ds_id}", per_page=500):
        key = (ds_id, r.name)
        ex = runs_by_name.get(key)
        if ex is None or quality(r) > quality(ex):
            runs_by_name[key] = r


# data[(row_idx, ds_id, K, short)] -> list of per-seed values
data = defaultdict(list)
missing = []
for row_idx, (label, kind, model, lm) in enumerate(ROWS):
    if kind == 'group':
        continue
    for ds_id, _ in DATASETS:
        for K in KS:
            name = run_name(kind, model, lm, K)
            run = runs_by_name.get((ds_id, name))
            if run is None or run.state != 'finished':
                missing.append((ds_id, name))
                continue
            for short, full, _ in METRICS:
                if run.summary.get(f'retrieval/avg_{full}') is None:
                    missing.append((ds_id, name + f' (no retrieval/{full})'))
                    continue
                for seed in range(N_SEEDS):
                    v = run.summary.get(f'retrieval/seed_{seed}/{full}')
                    if v is not None:
                        data[(row_idx, ds_id, K, short)].append(float(v))

print(f"\nCollected {len(data)} cells; expected {sum(1 for _, k, *_ in ROWS if k != 'group') * len(DATASETS) * len(KS) * len(METRICS)}")
short_cells = [(k, len(v)) for k, v in data.items() if len(v) != N_SEEDS]
if short_cells:
    print(f"WARNING: {len(short_cells)} cells have fewer than {N_SEEDS} per-seed values")
    for k, n in short_cells[:10]:
        print(f"  {k}: {n}/{N_SEEDS}")
if missing:
    # Dedupe and print first few
    seen = set()
    unique = []
    for m in missing:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    print(f"\nMissing runs (first 10): {unique[:10]}")


# Compute means + per-K Welch's t-test highlighting
means = {}
highlight = {}

for K in KS:
    for ds_id, _ in DATASETS:
        for short, _, _ in METRICS:
            per_row = {}
            for row_idx, (label, kind, *_) in enumerate(ROWS):
                if kind == 'group':
                    continue
                vs = data.get((row_idx, ds_id, K, short))
                if not vs:
                    continue
                per_row[row_idx] = np.array(vs)
            if not per_row:
                continue
            row_means = {ri: arr.mean() for ri, arr in per_row.items()}
            best_idx = max(row_means, key=row_means.get)
            best_vals = per_row[best_idx]
            for ri, arr in per_row.items():
                means[(ri, ds_id, K, short)] = row_means[ri]
                if ri == best_idx:
                    highlight[(ri, ds_id, K, short)] = True
                else:
                    t, p = stats.ttest_ind(arr, best_vals, equal_var=False)
                    highlight[(ri, ds_id, K, short)] = bool(p >= ALPHA)


def fmt(v):
    if v is None: return '---'
    return f".{int(round(v*1000)):03d}"


def cell(row_idx, ds_id, K, short):
    v = means.get((row_idx, ds_id, K, short))
    if v is None:
        return '---'
    txt = fmt(v)
    if highlight.get((row_idx, ds_id, K, short)):
        return rf'\colorbox{{lightblue}}{{\textbf{{{txt}}}}}'
    return txt


# Build LaTeX (table* with 7 columns: method + 2 metrics × 3 datasets)
out = []
out.append(r'\begin{table*}[t]')
out.append(r'\centering')
out.append(r'\begin{adjustbox}{width=\linewidth}')
out.append(r'\begin{tabular}{l *{6}{c}}')
out.append(r'\toprule')
out.append(r'\multicolumn{1}{c}{} &')
out.append(r'  \multicolumn{2}{c}{\large \textbf{20NewsGroup}} &')
out.append(r'  \multicolumn{2}{c}{\large \textbf{TweetTopic}} &')
out.append(r'  \multicolumn{2}{c}{\large \textbf{StackOverflow}} \\')
out.append(r'\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}')
out.append(r'Method &')
out.append(r'  P@5 & P@10 &')
out.append(r'  P@5 & P@10 &')
out.append(r'  P@5 & P@10 \\')
out.append(r'\midrule')

for K_idx, K in enumerate(KS):
    if K_idx > 0:
        out.append(r'\midrule')
    out.append(rf'\multicolumn{{7}}{{c}}{{\large \textbf{{K = {K}}}}} \\')
    out.append(r'\midrule')

    for row_idx, (label, kind, *_) in enumerate(ROWS):
        if kind == 'group':
            # Just an inline separator inside the K section; no extra midrule
            continue
        cells = []
        for ds_id, _ in DATASETS:
            for short, _, _ in METRICS:
                cells.append(cell(row_idx, ds_id, K, short))
        out.append(f"{label} & " + " & ".join(cells) + r" \\")

out.append(r'\bottomrule')
out.append(r'\end{tabular}')
out.append(r'\end{adjustbox}')
out.append(r'\caption{')
out.append(r'Complete retrieval evaluation (P@5, P@10) for each number of topics ($K=25, 50, 75, 100$), '
           r'each cell averaged over 5 random seeds.')
out.append(r"For each $(K, \text{dataset}, \text{metric})$ column, methods are compared using Welch's "
           r"independent samples $t$-test ($\alpha=0.05$). The best result and those not statistically "
           r'significantly different from it ($p \ge 0.05$) are \colorbox{lightblue}{\textbf{highlighted in blue}}.')
out.append(r'Rows without a suffix use the ProdLDA backbone with our proposed method; rows suffixed with '
           r'\texttt{+ ECRTM} or \texttt{+ FASTopic} use the ECRTM or FASTopic backbone.')
out.append(r'}')
out.append(r'\label{tab:complete-retrieval-results}')
out.append(r'\end{table*}')

latex = '\n'.join(out) + '\n'
for path in ['/home/toolkit/LLM-Topic-Modeling/table_retrieval_complete.md',
             '/home/toolkit/LLM-Topic-Modeling/table_retrieval_complete.tex']:
    with open(path, 'w') as f:
        f.write(latex)
    print(f"Wrote {path}")

# Per-K highlight summary
print(f"\n--- Per-K Ours-row highlight counts (out of 6 cells per row per K) ---")
for K in KS:
    print(f"\nK = {K}")
    for row_idx, (label, kind, *_) in enumerate(ROWS):
        if kind != 'ours':
            continue
        hl = sum(1 for ds_id, _ in DATASETS for short, *_ in METRICS
                 if highlight.get((row_idx, ds_id, K, short)))
        print(f"  {label:32s} {hl}/6")
