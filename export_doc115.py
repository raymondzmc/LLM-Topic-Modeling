"""Export doc #115 from 20 Newsgroups: document text + BoW and DST distributions.

Produces three artifacts in figures/targets/doc115/:
  - document.txt     : raw original document text
  - bow.json         : top-N (word, probability) pairs for the BoW target
  - dst.json         : top-N (word, probability) pairs for our DST target
                       (LLM softmax over top-k logits, temperature T)
  - prompt.md        : single human/LLM-readable file combining all three —
                       drop this into your image-gen model prompt.
"""
import os
import re
import json
import argparse
import numpy as np
from collections import Counter
from datasets import load_from_disk


def softmax(x, temp=1.0):
    z = (x - x.max()) / temp
    e = np.exp(z)
    return e / e.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, default=1701)
    parser.add_argument('--llm', default='ERNIE-4.5-0.3B-PT')
    parser.add_argument('--topk', type=int, default=20,
                        help='top-k mask on logits before softmax (DST)')
    parser.add_argument('--temperature', type=float, default=3.0)
    parser.add_argument('--top_n', type=int, default=15,
                        help='top-N words to export per distribution')
    parser.add_argument('--out_dir', default=None,
                        help='Defaults to figures/targets/doc<idx>')
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = f'figures/targets/doc{args.idx}'

    base = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(
        base, 'data', 'processed_data',
        f'20_newsgroups_{args.llm}_vocab_2000_last',
    )
    print(f"Loading {data_path}")
    dataset = load_from_disk(data_path)
    with open(os.path.join(data_path, 'vocab.json')) as f:
        vocab = json.load(f)
    token2idx = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)

    item = dataset[args.idx]
    label = item.get('label')
    ctx = item.get('context', '')
    m = re.search(r'Document:\s*', ctx)
    doc_text = (ctx[m.end():] if m else ctx).strip()
    bow_str = item['bow']
    logits = np.asarray(item['next_word_logits'], dtype=np.float64)

    # BoW distribution (normalized counts over the 2000-word vocab)
    counts = Counter(bow_str.split())
    p_bow = np.zeros(V, dtype=np.float64)
    for w, c in counts.items():
        if w in token2idx:
            p_bow[token2idx[w]] = c
    if p_bow.sum() > 0:
        p_bow /= p_bow.sum()

    # DST: top-k mask + temperature softmax (matches generative_fastopic default)
    keep = np.argpartition(-logits, args.topk - 1)[:args.topk]
    masked = np.full_like(logits, -np.inf)
    masked[keep] = logits[keep]
    p_dst = softmax(masked, temp=args.temperature)

    def top(p):
        idx = np.argsort(p)[::-1][:args.top_n]
        return [(vocab[i], float(p[i])) for i in idx if p[i] > 0]

    bow_top = top(p_bow)
    dst_top = top(p_dst)

    top_bow_set = set(w for w, _ in bow_top)
    top_dst_set = set(w for w, _ in dst_top)
    iou = (len(top_bow_set & top_dst_set)
           / max(len(top_bow_set | top_dst_set), 1))

    # Write outputs
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'document.txt'), 'w') as f:
        f.write(doc_text + '\n')
    with open(os.path.join(args.out_dir, 'bow.json'), 'w') as f:
        json.dump(bow_top, f, indent=2)
    with open(os.path.join(args.out_dir, 'dst.json'), 'w') as f:
        json.dump(dst_top, f, indent=2)

    # Markdown prompt — easy to paste into Claude / GPT / Gemini for figure-gen
    prompt = []
    label_names = [
        'alt.atheism','comp.graphics','comp.os.ms-windows.misc',
        'comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','comp.windows.x',
        'misc.forsale','rec.autos','rec.motorcycles','rec.sport.baseball',
        'rec.sport.hockey','sci.crypt','sci.electronics','sci.med','sci.space',
        'soc.religion.christian','talk.politics.guns','talk.politics.mideast',
        'talk.politics.misc','talk.religion.misc',
    ]
    label_name = (label_names[int(label)]
                  if isinstance(label, int) and 0 <= label < len(label_names)
                  else str(label))
    prompt.append(f"# 20 Newsgroups doc #{args.idx} — figure-source data\n")
    prompt.append(f"- LLM: {args.llm}")
    prompt.append(f"- DST: top-k={args.topk}, T={args.temperature}")
    prompt.append(f"- BoW vocab size: {V}")
    if label is not None and label != '':
        prompt.append(f"- Label: {label}  ({label_name})")
    prompt.append(f"- Top-{args.top_n} IoU(BoW, DST): {iou:.2f}\n")
    prompt.append("## Document\n")
    prompt.append("```")
    prompt.append(doc_text)
    prompt.append("```\n")
    prompt.append(f"## BoW target — top {args.top_n} words")
    prompt.append("| Rank | Word | Probability |")
    prompt.append("|---:|:---|---:|")
    for r, (w, p) in enumerate(bow_top, 1):
        prompt.append(f"| {r} | `{w}` | {p:.4f} |")
    prompt.append("")
    prompt.append(f"## DST target (Ours) — top {args.top_n} words")
    prompt.append("| Rank | Word | Probability |")
    prompt.append("|---:|:---|---:|")
    for r, (w, p) in enumerate(dst_top, 1):
        prompt.append(f"| {r} | `{w}` | {p:.4f} |")
    prompt.append("")

    md = '\n'.join(prompt)
    with open(os.path.join(args.out_dir, 'prompt.md'), 'w') as f:
        f.write(md)

    print(md)
    print(f"\nWrote: document.txt, bow.json, dst.json, prompt.md  ->  {args.out_dir}")


if __name__ == '__main__':
    main()
