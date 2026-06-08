"""Validate processed topic modeling datasets before training.

Checks that hidden states (input_embeddings) and projected vocab distributions
(next_word_logits) are well-formed and numerically valid. Exits with code 1 on
any failure so it can gate the training step in a pipeline.
"""

import os
import sys
import json
import argparse

import numpy as np
from scipy.special import softmax, logsumexp
from datasets import load_from_disk


def validate_dataset(dataset_path: str, verbose: bool = True) -> bool:
    """Run all validation checks on a single processed dataset.

    Returns True if the dataset passes all checks, False otherwise.
    """
    name = os.path.basename(dataset_path)
    print(f"\n{'='*70}")
    print(f"Validating: {name}")
    print(f"  Path: {dataset_path}")
    print(f"{'='*70}")
    errors = []

    # --- 1. Check required files ---
    vocab_path = os.path.join(dataset_path, "vocab.json")
    metadata_path = os.path.join(dataset_path, "metadata.json")

    if not os.path.exists(vocab_path):
        errors.append("vocab.json missing")
    if not os.path.exists(metadata_path):
        errors.append("metadata.json missing")

    if errors:
        for e in errors:
            print(f"  FAIL: {e}")
        return False

    with open(vocab_path) as f:
        vocab = json.load(f)
    with open(metadata_path) as f:
        metadata = json.load(f)

    vocab_size = len(vocab)
    print(f"  Vocab size: {vocab_size}")
    print(f"  Model: {metadata.get('args', {}).get('model_name', 'unknown')}")

    if vocab_size == 0:
        errors.append("vocab.json is empty")
        for e in errors:
            print(f"  FAIL: {e}")
        return False

    # --- 2. Load dataset ---
    try:
        dataset = load_from_disk(dataset_path)
    except Exception as exc:
        errors.append(f"Failed to load dataset: {exc}")
        for e in errors:
            print(f"  FAIL: {e}")
        return False

    n_samples = len(dataset)
    print(f"  Samples: {n_samples}")

    required_cols = {"input_embeddings", "next_word_logits", "bow"}
    missing_cols = required_cols - set(dataset.column_names)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
        for e in errors:
            print(f"  FAIL: {e}")
        return False

    # --- 3. Per-sample validation (stream through dataset) ---
    all_logit_mins = []
    all_logit_maxs = []
    all_logit_means = []
    all_entropies = []
    emb_dim = None
    n_zero_bow = 0
    n_bad_embeddings = 0
    n_bad_logits = 0
    n_nan_inf_emb = 0
    n_nan_inf_logits = 0
    n_wrong_logit_len = 0
    n_wrong_emb_dim = 0

    for i, sample in enumerate(dataset):
        # -- Embeddings --
        emb = np.array(sample["input_embeddings"], dtype=np.float64)

        if emb.ndim == 2:
            emb = emb[-1]

        if emb_dim is None:
            emb_dim = emb.shape[0]
            print(f"  Embedding dim: {emb_dim}")
        elif emb.shape[0] != emb_dim:
            n_wrong_emb_dim += 1

        if not np.all(np.isfinite(emb)):
            n_nan_inf_emb += 1

        if np.allclose(emb, 0.0):
            n_bad_embeddings += 1

        # -- Next-word logits --
        logits = np.array(sample["next_word_logits"], dtype=np.float64)

        if logits.shape[0] != vocab_size:
            n_wrong_logit_len += 1
            continue

        if not np.all(np.isfinite(logits)):
            n_nan_inf_logits += 1
            continue

        probs = softmax(logits)

        if not np.all(np.isfinite(probs)):
            n_bad_logits += 1
            continue

        prob_sum = probs.sum()
        if not np.isclose(prob_sum, 1.0, atol=1e-4):
            n_bad_logits += 1

        entropy = -np.sum(probs * np.log(probs + 1e-12))

        all_logit_mins.append(logits.min())
        all_logit_maxs.append(logits.max())
        all_logit_means.append(logits.mean())
        all_entropies.append(entropy)

        # -- BoW --
        bow = sample["bow"]
        if not bow or not bow.strip():
            n_zero_bow += 1

    # --- 4. Report ---
    print(f"\n  --- Embedding Checks ---")
    print(f"  All-zero embeddings:       {n_bad_embeddings}/{n_samples}")
    print(f"  NaN/Inf embeddings:        {n_nan_inf_emb}/{n_samples}")
    print(f"  Wrong embedding dim:       {n_wrong_emb_dim}/{n_samples}")

    print(f"\n  --- Logit / Distribution Checks ---")
    print(f"  Wrong logit length:        {n_wrong_logit_len}/{n_samples}")
    print(f"  NaN/Inf logits:            {n_nan_inf_logits}/{n_samples}")
    print(f"  Invalid softmax dist:      {n_bad_logits}/{n_samples}")
    print(f"  Empty BoW:                 {n_zero_bow}/{n_samples}")

    if all_logit_mins:
        print(f"\n  --- Distribution Statistics ---")
        print(f"  Logit min  (mean/min/max): {np.mean(all_logit_mins):+.4f} / {np.min(all_logit_mins):+.4f} / {np.max(all_logit_mins):+.4f}")
        print(f"  Logit max  (mean/min/max): {np.mean(all_logit_maxs):+.4f} / {np.min(all_logit_maxs):+.4f} / {np.max(all_logit_maxs):+.4f}")
        print(f"  Logit mean (mean/min/max): {np.mean(all_logit_means):+.4f} / {np.min(all_logit_means):+.4f} / {np.max(all_logit_means):+.4f}")
        print(f"  Entropy    (mean/min/max): {np.mean(all_entropies):.4f} / {np.min(all_entropies):.4f} / {np.max(all_entropies):.4f}")
        print(f"  Max possible entropy:      {np.log(vocab_size):.4f}")

    # --- 5. Determine pass/fail ---
    if n_nan_inf_emb > 0:
        errors.append(f"{n_nan_inf_emb} samples have NaN/Inf embeddings")
    if n_nan_inf_logits > 0:
        errors.append(f"{n_nan_inf_logits} samples have NaN/Inf logits")
    if n_wrong_logit_len > 0:
        errors.append(f"{n_wrong_logit_len} samples have wrong logit length (expected {vocab_size})")
    if n_wrong_emb_dim > 0:
        errors.append(f"{n_wrong_emb_dim} samples have inconsistent embedding dim")
    if n_bad_logits > 0:
        errors.append(f"{n_bad_logits} samples have invalid softmax distributions")
    if n_bad_embeddings == n_samples:
        errors.append("All embeddings are zero vectors")

    if errors:
        print(f"\n  RESULT: FAILED")
        for e in errors:
            print(f"    - {e}")
        return False
    else:
        print(f"\n  RESULT: PASSED")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Validate processed topic modeling datasets"
    )
    parser.add_argument(
        "dataset_paths",
        nargs="+",
        help="Paths to processed dataset directories",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
    )
    args = parser.parse_args()

    all_passed = True
    results = {}

    for path in args.dataset_paths:
        if not os.path.isdir(path):
            print(f"\nERROR: {path} does not exist or is not a directory")
            results[path] = False
            all_passed = False
            continue
        passed = validate_dataset(path, verbose=args.verbose)
        results[path] = passed
        if not passed:
            all_passed = False

    # Summary
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    for path, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {os.path.basename(path)}")

    if all_passed:
        print(f"\nAll {len(results)} datasets passed validation.")
        sys.exit(0)
    else:
        n_failed = sum(1 for v in results.values() if not v)
        print(f"\n{n_failed}/{len(results)} datasets FAILED validation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
