"""Visualize the difference between BoW targets and LLM targets (post-softmax).

For each of the 3 datasets, picks 5 representative documents and produces a
side-by-side horizontal bar chart of the top-15 vocabulary words under each
target distribution.

Usage:
    python visualize_targets.py [--llm ERNIE-4.5-0.3B-PT]
                                [--topk 20] [--temperature 3.0]
                                [--out_dir figures/targets]
"""
import os
import re
import json
import argparse
import textwrap
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datasets import load_from_disk

DATASETS = ['20_newsgroups', 'tweet_topic', 'stackoverflow']
DATASET_LABELS = {
    '20_newsgroups': '20 Newsgroups',
    'tweet_topic': 'Tweet Topic',
    'stackoverflow': 'StackOverflow',
}
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'processed_data')


def softmax(x, temp=1.0):
    z = (x - x.max()) / temp
    e = np.exp(z)
    return e / e.sum()


def llm_target(logits, vocab_size, topk, temperature):
    """Top-k mask + temperature softmax over the BoW vocab."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape[0] != vocab_size:
        raise ValueError(f"logits dim {logits.shape[0]} != vocab size {vocab_size}")
    if topk is not None and topk < vocab_size:
        keep = np.argpartition(-logits, topk - 1)[:topk]
        masked = np.full_like(logits, -np.inf)
        masked[keep] = logits[keep]
    else:
        masked = logits
    return softmax(masked, temp=temperature)


def bow_target(bow_string, token2idx, vocab_size):
    """Normalized BoW frequency vector over the BoW vocab."""
    counts = Counter(bow_string.split())
    vec = np.zeros(vocab_size, dtype=np.float64)
    for tok, c in counts.items():
        if tok in token2idx:
            vec[token2idx[tok]] = c
    s = vec.sum()
    return vec / s if s > 0 else vec


def extract_document_text(context):
    """Strip the prompt template from a context string and return just the document.

    The processed dataset wraps each document in a prompt of the form:
        "Generate a label ...\n\nDocument: <text>"
    We extract everything after the last 'Document:' marker; if absent, fall
    back to the full context.
    """
    if not isinstance(context, str):
        return str(context)
    m = re.search(r'Document:\s*', context)
    return context[m.end():].strip() if m else context.strip()


def truncate_for_display(text, max_chars=600):
    text = re.sub(r'\s+', ' ', text).strip()
    return text if len(text) <= max_chars else text[:max_chars - 1].rstrip() + '…'


def pick_examples(dataset, n=5, seed=0, strategy='random',
                  topk=20, temperature=3.0, token2idx=None, vocab_size=None,
                  topn_iou=15):
    """Pick n example documents.

    Strategies:
      - 'random': uniform over candidates (medium-length docs)
      - 'low_iou': lowest top-N IoU between BoW and LLM softmax (max motivation)
      - 'label_diverse_low_iou': lowest IoU subject to one example per label
    """
    rng = np.random.default_rng(seed)
    bow_lens = np.array([len(b.split()) for b in dataset['bow']])
    # Drop docs with empty/near-empty BoW (degenerate — chart is meaningless)
    nonempty = bow_lens >= 5
    if not nonempty.any():
        raise ValueError("No documents with >= 5 BoW tokens")
    lo, hi = np.percentile(bow_lens[nonempty], [30, 80])
    candidates = np.where(nonempty & (bow_lens >= lo) & (bow_lens <= hi))[0]

    if strategy == 'random':
        return rng.choice(candidates, size=min(n, len(candidates)), replace=False)

    if token2idx is None or vocab_size is None:
        raise ValueError("low_iou strategies require token2idx and vocab_size")

    print(f"    computing IoU over {len(candidates)} candidate docs for "
          f"strategy={strategy}...")
    iou_scores = np.empty(len(candidates), dtype=np.float64)
    for j, idx in enumerate(candidates):
        item = dataset[int(idx)]
        p_bow = bow_target(item['bow'], token2idx, vocab_size)
        p_llm = llm_target(item['next_word_logits'], vocab_size, topk, temperature)
        a = set(np.argsort(p_bow)[::-1][:topn_iou])
        b = set(np.argsort(p_llm)[::-1][:topn_iou])
        iou_scores[j] = len(a & b) / max(len(a | b), 1)

    order = np.argsort(iou_scores)  # ascending = lowest IoU first

    if strategy == 'low_iou':
        return candidates[order[:n]]

    if strategy == 'label_diverse_low_iou':
        labels = dataset['label'] if 'label' in dataset.column_names else None
        if labels is None:
            return candidates[order[:n]]
        seen, picked = set(), []
        for j in order:
            idx = int(candidates[j])
            lab = labels[idx]
            if lab in seen:
                continue
            seen.add(lab)
            picked.append(idx)
            if len(picked) >= n:
                break
        # Top up with low-IoU regardless of label if not enough labels
        if len(picked) < n:
            for j in order:
                idx = int(candidates[j])
                if idx not in picked:
                    picked.append(idx)
                    if len(picked) >= n:
                        break
        return np.array(picked)

    raise ValueError(f"Unknown strategy: {strategy}")


def plot_one_example(ax_bow, ax_llm, p_bow, p_llm, vocab, top_n=15,
                     bow_color='#4C72B0', llm_color='#C44E52'):
    # Use the union of top_n indices from both distributions, ordered by joint mass
    idx_bow = np.argsort(p_bow)[::-1][:top_n]
    idx_llm = np.argsort(p_llm)[::-1][:top_n]

    # Plot BoW (left)
    words_bow = [vocab[i] for i in idx_bow]
    ax_bow.barh(range(top_n), p_bow[idx_bow][::-1], color=bow_color, alpha=0.85)
    ax_bow.set_yticks(range(top_n))
    ax_bow.set_yticklabels(words_bow[::-1], fontsize=8)
    ax_bow.invert_xaxis()
    ax_bow.tick_params(axis='x', labelsize=7)
    ax_bow.spines['top'].set_visible(False)
    ax_bow.spines['right'].set_visible(False)

    # Plot LLM (right)
    words_llm = [vocab[i] for i in idx_llm]
    ax_llm.barh(range(top_n), p_llm[idx_llm][::-1], color=llm_color, alpha=0.85)
    ax_llm.set_yticks(range(top_n))
    ax_llm.set_yticklabels(words_llm[::-1], fontsize=8)
    ax_llm.tick_params(axis='x', labelsize=7)
    ax_llm.spines['top'].set_visible(False)
    ax_llm.spines['right'].set_visible(False)


def visualize_dataset(dataset_name, llm, n_examples, topk, temperature, out_dir,
                       seed, strategy='random', out_suffix=''):
    folder = f"{dataset_name}_{llm}_vocab_2000_last"
    path = os.path.join(PROCESSED_DIR, folder)
    if not os.path.isdir(path):
        print(f"  [{dataset_name}] processed data not found: {path}; skipping")
        return

    print(f"  [{dataset_name}] loading {path}")
    dataset = load_from_disk(path)
    with open(os.path.join(path, 'vocab.json')) as f:
        vocab = json.load(f)
    token2idx = {t: i for i, t in enumerate(vocab)}
    vocab_size = len(vocab)

    indices = pick_examples(
        dataset, n=n_examples, seed=seed, strategy=strategy,
        topk=topk, temperature=temperature,
        token2idx=token2idx, vocab_size=vocab_size,
    )

    # Per example: a bordered panel with header (height 0.6) + text (height 2.0)
    # + charts (height 4.0). One outer GridSpec per example, with hspace large
    # enough for a visible gap between examples.
    rows_per_ex = 3
    height_ratios = [0.6, 2.0, 4.0]
    per_ex_height = sum(height_ratios) * 0.55  # inches per example
    fig = plt.figure(figsize=(11.5, per_ex_height * n_examples + 0.7))
    outer = GridSpec(
        nrows=n_examples, ncols=1,
        hspace=0.45, figure=fig,
    )

    topn = 15
    for row, idx in enumerate(indices):
        item = dataset[int(idx)]
        bow_str = item['bow']
        logits = item['next_word_logits']
        doc_text = truncate_for_display(
            extract_document_text(item.get('context', '')), max_chars=450,
        )
        p_bow = bow_target(bow_str, token2idx, vocab_size)
        p_llm = llm_target(logits, vocab_size, topk=topk, temperature=temperature)

        top_bow = set(np.argsort(p_bow)[::-1][:topn])
        top_llm = set(np.argsort(p_llm)[::-1][:topn])
        iou = len(top_bow & top_llm) / max(len(top_bow | top_llm), 1)

        # Inner GridSpec: header / text / charts
        inner = outer[row].subgridspec(
            nrows=rows_per_ex, ncols=2,
            height_ratios=height_ratios, hspace=0.15, wspace=0.22,
        )

        # Header (spans both columns) — bold, with metadata
        ax_hdr = fig.add_subplot(inner[0, :])
        ax_hdr.axis('off')
        label = item.get('label', '')
        header = f"doc #{idx}"
        if label != '' and label is not None:
            header += f"   label={label}"
        header += (f"   {len(bow_str.split())} BoW tokens"
                   f"   top-{topn} IoU = {iou:.2f}")
        ax_hdr.text(0.005, 0.5, header, fontsize=10, fontweight='bold',
                    family='sans-serif', va='center', ha='left',
                    transform=ax_hdr.transAxes,
                    bbox=dict(boxstyle='round,pad=0.35',
                              facecolor='#EDEDED', edgecolor='#888888',
                              linewidth=0.6))

        # Document text axis — clipped so overflow can't bleed into next example
        ax_text = fig.add_subplot(inner[1, :])
        ax_text.set_xticks([]); ax_text.set_yticks([])
        for s in ax_text.spines.values():
            s.set_edgecolor('#CCCCCC'); s.set_linewidth(0.7)
        ax_text.set_facecolor('#FAFAFA')
        wrapped = '\n'.join(textwrap.wrap(doc_text, width=140)) or '(empty)'
        ax_text.text(0.01, 0.97, wrapped, fontsize=8, family='monospace',
                     va='top', ha='left', transform=ax_text.transAxes,
                     color='#222222', clip_on=True, wrap=True)

        # BoW and LLM charts
        ax_bow = fig.add_subplot(inner[2, 0])
        ax_llm = fig.add_subplot(inner[2, 1])
        plot_one_example(ax_bow, ax_llm, p_bow, p_llm, vocab, top_n=topn)
        ax_bow.set_title("BoW target", fontsize=9, loc='right')
        ax_llm.set_title("LLM softmax target", fontsize=9, loc='left')

    fig.suptitle(
        f"{DATASET_LABELS[dataset_name]}  ·  LLM = {llm}  ·  "
        f"top-k={topk}, T={temperature}",
        fontsize=12, y=0.995,
    )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_name}_targets{out_suffix}.png")
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  [{dataset_name}] saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llm', default='ERNIE-4.5-0.3B-PT',
                        help='LLM suffix used for processed_data folder name')
    parser.add_argument('--topk', type=int, default=20,
                        help='Top-k mask applied to LLM logits before softmax')
    parser.add_argument('--temperature', type=float, default=3.0,
                        help='Softmax temperature applied to LLM logits')
    parser.add_argument('--n_examples', type=int, default=5,
                        help='Documents to visualize per dataset')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for example selection (reproducibility)')
    parser.add_argument('--out_dir', default='figures/targets',
                        help='Output directory for PNGs')
    parser.add_argument('--datasets', default=','.join(DATASETS),
                        help='Comma-separated datasets to render '
                             f'(any of {DATASETS})')
    parser.add_argument('--strategy', default='random',
                        choices=['random', 'low_iou', 'label_diverse_low_iou'],
                        help='Example-selection strategy')
    parser.add_argument('--out_suffix', default='',
                        help='Suffix appended to output filenames')
    args = parser.parse_args()

    requested = [d.strip() for d in args.datasets.split(',') if d.strip()]
    bad = [d for d in requested if d not in DATASETS]
    if bad:
        parser.error(f"Unknown datasets: {bad}; choose from {DATASETS}")

    print(f"LLM={args.llm}  topk={args.topk}  T={args.temperature}  "
          f"n={args.n_examples}  strategy={args.strategy}  "
          f"datasets={requested}  out={args.out_dir}")
    for d in requested:
        visualize_dataset(d, args.llm, args.n_examples, args.topk,
                          args.temperature, args.out_dir, args.seed,
                          strategy=args.strategy, out_suffix=args.out_suffix)


if __name__ == '__main__':
    main()
