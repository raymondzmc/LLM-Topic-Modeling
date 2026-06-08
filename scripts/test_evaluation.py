"""End-to-end smoke test for all evaluation metrics.

Downloads a small existing processed dataset, trains a quick GenerativeTM,
and runs the full evaluate_topic_model() pipeline to verify all 12 metrics
produce valid output.
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loaders import load_training_data
from data.dataset.ctm_dataset import get_ctm_dataset_from_processed_data
from models.ctm import GenerativeTM
from evaluation.metrics import evaluate_topic_model
from utils.embeddings import get_openai_embedding

EXPECTED_METRICS = {
    "topic_diversity",
    "inverted_rbo",
    "purity",
    "inverse_purity",
    "harmonic_purity",
    "ari",
    "mis",
    "npmi",
    "cv",
    "cv_wiki",
    "llm_rating",
    "training_time",
}

DATA_PATH = "raymondzmc/tweet_topic_ERNIE-4.5-0.3B-PT_vocab_2000_last"
NUM_TOPICS = 5
NUM_EPOCHS = 10
TOP_WORDS = 15


def main():
    print("=" * 60)
    print("Evaluation Metrics Smoke Test")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/5] Loading dataset...")
    training_data = load_training_data(DATA_PATH, for_generative=True)
    vocab = training_data.vocab
    labels = training_data.labels
    bow_corpus = training_data.bow_corpus

    print(f"  Dataset size: {len(training_data.processed_dataset)}")
    print(f"  Vocab size: {len(vocab)}")
    print(f"  Has labels: {labels is not None}")
    print(f"  BOW corpus size: {len(bow_corpus)}")

    # Step 2: Prepare CTM dataset
    print("\n[2/5] Preparing CTM dataset...")
    ctm_dataset = get_ctm_dataset_from_processed_data(
        training_data.processed_dataset,
        vocab,
    )

    # Step 3: Train model
    print(f"\n[3/5] Training GenerativeTM (K={NUM_TOPICS}, epochs={NUM_EPOCHS})...")
    np.random.seed(42)
    import torch
    torch.manual_seed(42)

    model = GenerativeTM(
        vocab_size=len(vocab),
        embedding_size=ctm_dataset.x_embeddings.shape[1],
        num_topics=NUM_TOPICS,
        num_epochs=NUM_EPOCHS,
        top_words=TOP_WORDS,
    )
    model.fit(ctm_dataset)
    model_output = model.get_info()

    print(f"  Topics generated: {len(model_output['topics'])}")
    print(f"  Words per topic: {len(model_output['topics'][0])}")
    print(f"  topic-document-matrix shape: {model_output['topic-document-matrix'].shape}")

    # Step 4: Load/compute vocab embeddings
    print("\n[4/5] Loading vocab embeddings...")
    vocab_embedding_path = os.path.join(training_data.local_path, "vocab_embeddings.json")
    if os.path.exists(vocab_embedding_path):
        with open(vocab_embedding_path, encoding="utf-8") as f:
            vocab_embeddings = json.load(f)
        print(f"  Loaded from cache ({len(vocab_embeddings)} words)")
    else:
        print("  Computing OpenAI embeddings for vocab...")
        vocab_embeddings = get_openai_embedding(vocab)
        os.makedirs(os.path.dirname(vocab_embedding_path), exist_ok=True)
        with open(vocab_embedding_path, "w", encoding="utf-8") as f:
            json.dump(vocab_embeddings, f)
        print(f"  Computed and cached ({len(vocab_embeddings)} words)")

    # Step 5: Run evaluation
    print(f"\n[5/5] Running evaluate_topic_model()...")
    model_output["training_time"] = 0.0
    results = evaluate_topic_model(
        model_output,
        top_words=TOP_WORDS,
        test_corpus=bow_corpus,
        embeddings=vocab_embeddings,
        labels=labels,
    )
    results["training_time"] = model_output["training_time"]

    # Verify results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for key in sorted(EXPECTED_METRICS):
        value = results.get(key)
        status = "OK" if value is not None else "MISSING"
        print(f"  {key:25s} = {value!s:>15s}  [{status}]")

    missing = EXPECTED_METRICS - set(results.keys())
    extra = set(results.keys()) - EXPECTED_METRICS

    print("\n" + "-" * 60)
    if missing:
        print(f"MISSING metrics: {missing}")
    if extra:
        print(f"Extra metrics (unexpected): {extra}")

    if not missing:
        print("ALL 12 EXPECTED METRICS PRESENT -- PASS")
        return 0
    else:
        print(f"FAIL -- {len(missing)} metrics missing")
        return 1


if __name__ == "__main__":
    sys.exit(main())
