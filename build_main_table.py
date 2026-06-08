"""Build the camera-ready DSL results table with Welch's t-test highlighting.

For each (dataset, metric) column:
  1. Collect 20 per-seed-per-K measurements for each method.
  2. Find the method with the highest mean (since all metrics are higher-better).
  3. Welch's t-test each other method's 20 values vs the best's 20 values.
  4. Highlight (best or p >= 0.05) cells in blue.

Writes table_main.tex (LaTeX, camera-ready) and prints diagnostics.
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
DATASETS = [('20_newsgroups', '20NewsGroup'),
            ('tweet_topic',   'TweetTopic'),
            ('stackoverflow', 'StackOverflow')]
KS = [25, 50, 75, 100]
N_SEEDS = 5
ALPHA = 0.05

# Row schema: (display_label, kind, model_id, lm_id)
# kind: 'baseline' uses run name = "{model}_K{K}"
# kind: 'ours'     uses run name = "{model}_{lm}_K{K}"
ROWS = [
    ('LDA',         'baseline', 'lda',      None),
    ('ProdLDA',     'baseline', 'prodlda',  None),
    ('CombinedTM',  'baseline', 'combined', None),
    ('ZeroshotTM',  'baseline', 'zeroshot', None),
    ('ETM',         'baseline', 'etm',      None),
    ('BERTopic',    'baseline', 'bertopic', None),
    ('ECRTM',       'baseline', 'ecrtm',    None),
    ('FASTopic',    'baseline', 'fastopic', None),
    # Ours: ProdLDA + DSL — ordered by LM parameter count
    ('PRODLDA_GROUP', 'group',  None,       None),
    ('ERNIE-4.5-0.3B', 'ours', 'generative', 'ERNIE-4.5-0.3B-PT'),
    ('Qwen3.5-0.8B',   'ours', 'generative', 'Qwen3.5-0.8B'),
    ('Llama-3.2-1B',   'ours', 'generative', 'Llama-3.2-1B-Instruct'),
    ('Phi-3-mini',     'ours', 'generative', 'Phi-3-mini-128k-instruct'),
    ('Llama-3.1-8B',   'ours', 'generative', 'Llama-3.1-8B-Instruct'),
    # Ours: FASTopic + DSL
    ('FASTOPIC_GROUP', 'group', None,       None),
    ('ERNIE-4.5-0.3B', 'ours', 'generative_fastopic', 'ERNIE-4.5-0.3B-PT'),
    ('Qwen3.5-0.8B',   'ours', 'generative_fastopic', 'Qwen3.5-0.8B'),
    ('Llama-3.2-1B',   'ours', 'generative_fastopic', 'Llama-3.2-1B-Instruct'),
    ('Phi-3-mini',     'ours', 'generative_fastopic', 'Phi-3-mini-128k-instruct'),
    ('Llama-3.1-8B',   'ours', 'generative_fastopic', 'Llama-3.1-8B-Instruct'),
    # Ours: ECRTM + DSL
    ('ECRTM_GROUP',  'group',  None,       None),
    ('ERNIE-4.5-0.3B', 'ours', 'generative_ecrtm', 'ERNIE-4.5-0.3B-PT'),
    ('Qwen3.5-0.8B',   'ours', 'generative_ecrtm', 'Qwen3.5-0.8B'),
    ('Llama-3.2-1B',   'ours', 'generative_ecrtm', 'Llama-3.2-1B-Instruct'),
    ('Phi-3-mini',     'ours', 'generative_ecrtm', 'Phi-3-mini-128k-instruct'),
    ('Llama-3.1-8B',   'ours', 'generative_ecrtm', 'Llama-3.1-8B-Instruct'),
]

METRICS = [
    ('cv',     'cv_wiki',      r'$C_V$'),
    ('llm',    'llm_rating',   'LLM'),
    ('irbo',   'inverted_rbo', 'I-RBO'),
    ('purity', 'purity',       'Purity'),
]


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
    return (non_degenerate, finished, has_cv, ari)


# Fetch all runs once per project
api = wandb.Api(timeout=60)
runs_by_name = {}  # (dataset_id, name) -> best run
for ds_id, _ in DATASETS:
    print(f"Fetching {ds_id}...", file=sys.stderr)
    for r in api.runs(f"{ENTITY}/{ds_id}", per_page=500):
        key = (ds_id, r.name)
        ex = runs_by_name.get(key)
        if ex is None or quality(r) > quality(ex):
            runs_by_name[key] = r


# Collect per-seed-per-K values: data[(row_idx, ds_id, metric_short)] = list[20]
data = defaultdict(list)
problems = []
for row_idx, (label, kind, model, lm) in enumerate(ROWS):
    if kind == 'group':
        continue
    for ds_id, _ in DATASETS:
        for K in KS:
            name = run_name(kind, model, lm, K)
            run = runs_by_name.get((ds_id, name))
            if run is None or run.state != 'finished':
                problems.append((ds_id, name, 'missing/unfinished'))
                continue
            for short, full, _ in METRICS:
                for seed in range(N_SEEDS):
                    v = run.summary.get(f'seed_{seed}/{full}')
                    if v is None:
                        problems.append((ds_id, name, f'seed_{seed}/{full} missing'))
                        continue
                    data[(row_idx, ds_id, short)].append(float(v))


# Sanity-check sample sizes
print(f"\nCollected {len(data)} cells. Expected: {sum(1 for _, k, *_ in ROWS if k != 'group') * len(DATASETS) * len(METRICS)}")
short_cells = []
for key, values in data.items():
    if len(values) != N_SEEDS * len(KS):
        short_cells.append((key, len(values)))
if short_cells:
    print(f"WARNING: {len(short_cells)} cells have fewer than {N_SEEDS*len(KS)} values:")
    for key, n in short_cells[:10]:
        print(f"  {key}: {n}")
if problems:
    print(f"\n{len(problems)} problems encountered; first 5:")
    for p in problems[:5]:
        print(f"  {p}")


# Compute means and significance per (ds, metric)
means = {}      # (row_idx, ds_id, short) -> mean
highlight = {}  # (row_idx, ds_id, short) -> bool

for ds_id, _ in DATASETS:
    for short, _, _ in METRICS:
        # Gather all row means
        per_row = {}
        for row_idx, (label, kind, *_) in enumerate(ROWS):
            if kind == 'group':
                continue
            vals = data.get((row_idx, ds_id, short))
            if not vals:
                continue
            per_row[row_idx] = np.array(vals)
        if not per_row:
            continue
        # Best mean (higher = better for all 4 metrics)
        means_per_row = {ri: arr.mean() for ri, arr in per_row.items()}
        best_idx = max(means_per_row, key=means_per_row.get)
        best_vals = per_row[best_idx]
        for ri, arr in per_row.items():
            means[(ri, ds_id, short)] = means_per_row[ri]
            if ri == best_idx:
                highlight[(ri, ds_id, short)] = True
            else:
                # Welch's t-test
                t, p = stats.ttest_ind(arr, best_vals, equal_var=False)
                highlight[(ri, ds_id, short)] = bool(p >= ALPHA)


# Formatting helpers
def fmt(v, short):
    if v is None:
        return '---'
    if short == 'llm':
        return f"{v:.2f}"
    if short == 'irbo':
        if v >= 0.9995:
            return "1.000"
        return f".{int(round(v*1000)):03d}"
    # cv, purity
    return f".{int(round(v*1000)):03d}"


def cell(row_idx, ds_id, short):
    v = means.get((row_idx, ds_id, short))
    if v is None:
        return '---'
    txt = fmt(v, short)
    if highlight.get((row_idx, ds_id, short)):
        return rf'\colorbox{{lightblue}}{{\textbf{{{txt}}}}}'
    return txt


# Build LaTeX
out = []
out.append(r'\begin{table*}[ht]')
out.append(r'\centering')
out.append(r'\begin{adjustbox}{width=\linewidth}')
out.append(r'\begin{tabular}{l *{12}{c}}')
out.append(r'\toprule')
out.append(r'\multicolumn{1}{c}{} &')
out.append(r'  \multicolumn{4}{c}{\large \textbf{20NewsGroup}} &')
out.append(r'  \multicolumn{4}{c}{\large \textbf{TweetTopic}} &')
out.append(r'  \multicolumn{4}{c}{\large \textbf{StackOverflow}} \\')
out.append(r'\cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13}')
out.append(r'\multicolumn{1}{c}{} &')
headers = []
for _ in DATASETS:
    for _, _, disp in METRICS:
        headers.append(rf'  \texttt{{\small {disp}}}')
out.append(' &\n'.join(headers) + r' \\')
out.append(r'\midrule')

group_labels = {
    'PRODLDA_GROUP':  r'\multicolumn{13}{c}{\large \textit{Ours: ProdLDA + DSL}} \\',
    'FASTOPIC_GROUP': r'\multicolumn{13}{c}{\large \textit{Ours: FASTopic + DSL}} \\',
    'ECRTM_GROUP':    r'\multicolumn{13}{c}{\large \textit{Ours: ECRTM + DSL}} \\',
}

for row_idx, (label, kind, *_) in enumerate(ROWS):
    if kind == 'group':
        out.append(r'\midrule')
        out.append(group_labels[label])
        out.append(r'\midrule')
        continue

    cells = []
    for ds_id, _ in DATASETS:
        for short, _, _ in METRICS:
            cells.append(cell(row_idx, ds_id, short))

    if kind == 'ours':
        row_label = rf'\texttt{{{label}}}'
    else:
        row_label = label

    out.append(row_label + ' &\n' + ' & '.join(cells) + r' \\')

out.append(r'\bottomrule')
out.append(r'\end{tabular}')
out.append(r'\end{adjustbox}')
out.append(r'\caption{')
out.append(r'Automatic evaluation results on the top-15 words averaged over 4 numbers of topics ($K=25, 50, 75, 100$), '
           r'where results for each $K$ are averaged over 5 random seeds (20 measurements per cell).')
out.append(r"For each dataset and metric, methods are compared using Welch's independent samples $t$-test "
           r'\citep{welch1947generalization} with $\alpha=0.05$.')
out.append(r'The best-performing method, as well as results that are not statistically significantly different from it '
           r'($p \ge 0.05$), are \colorbox{lightblue}{\textbf{highlighted in blue}}.')
out.append(r'}')
out.append(r'\label{tab:main-results}')
out.append(r'\vspace{-1em}')
out.append(r'\end{table*}')

latex = '\n'.join(out) + '\n'
out_path = '/home/toolkit/LLM-Topic-Modeling/table_main.tex'
with open(out_path, 'w') as f:
    f.write(latex)
print(f"\nWrote {out_path}")
print(f"\n--- Significance summary (highlighted cells per row, max 12) ---")
for row_idx, (label, kind, *_) in enumerate(ROWS):
    if kind == 'group':
        continue
    hl = sum(1 for ds_id, _ in DATASETS for short, *_ in METRICS
             if highlight.get((row_idx, ds_id, short)))
    pre = '  ' if kind == 'ours' else ''
    print(f"{pre}{label:25s} {hl}/12 highlighted")
