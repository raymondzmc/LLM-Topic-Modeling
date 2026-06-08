"""
Robust TopicGPT assignment+correction+evaluation for 20newsgroups.
Saves progress incrementally and handles API timeouts.

Usage: python scripts/run_topicgpt_20news_assign.py
"""

import os
import sys
import re
import json
import signal
import pandas as pd
from tqdm import trange
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

TOPICGPT_DIR = os.path.join(PROJECT_ROOT, "topicGPT")
sys.path.insert(0, TOPICGPT_DIR)
os.chdir(TOPICGPT_DIR)

from topicgpt_python.utils import APIClient, TopicTree, calculate_purity, calculate_metrics
from topicgpt_python import correct_topics

API = "openai"
MODEL = "gpt-4o-mini"

DATA_FILE = "data/input/20newsgroups.jsonl"
TOPIC_FILE = "data/output/20newsgroups/refinement.md"
ASSIGNMENT_OUTPUT = "data/output/20newsgroups/assignment.jsonl"
CORRECTION_OUTPUT = "data/output/20newsgroups/assignment_corrected.jsonl"
PROGRESS_FILE = "data/output/20newsgroups/assignment_progress.jsonl"

TIMEOUT_SECONDS = 30


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("API call timed out")


def robust_assign():
    print("\n" + "=" * 60)
    print("STEP 1: Robust Topic Assignment (full dataset)")
    print("=" * 60)

    api_client = APIClient(api=API, model=MODEL)
    max_tokens, temperature, top_p = 1000, 0.0, 1.0

    context = 128000
    context_len = context - max_tokens

    df = pd.read_json(DATA_FILE, lines=True)
    docs = df["text"].tolist()
    assignment_prompt = open("prompt/assignment.txt", "r").read()
    topics_root = TopicTree().from_topic_list(TOPIC_FILE, from_file=True)
    tree_str = "\n".join(topics_root.to_topic_list(desc=True, count=False))

    print(f"Total docs: {len(docs)}")
    print(f"Using topics from: {TOPIC_FILE}")

    # Load existing progress
    completed = {}
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            for line in f:
                rec = json.loads(line)
                completed[rec["idx"]] = rec["response"]
        print(f"Resuming from {len(completed)} completed assignments")

    progress_f = open(PROGRESS_FILE, "a")

    for i in trange(len(docs), desc="Assigning"):
        if i in completed:
            continue

        doc = docs[i]

        # Truncate doc if needed
        max_doc_len = (
            context_len
            - api_client.estimate_token_count(assignment_prompt)
            - api_client.estimate_token_count(tree_str)
        )
        if api_client.estimate_token_count(doc) > max_doc_len:
            doc = api_client.truncating(doc, max_doc_len)

        prompt = assignment_prompt.format(Document=doc, tree=tree_str)

        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(TIMEOUT_SECONDS)
            response = api_client.iterative_prompt(
                prompt, max_tokens, temperature, top_p=top_p, num_try=2
            )
            signal.alarm(0)
        except (TimeoutError, Exception) as e:
            signal.alarm(0)
            response = "Error"
            print(f"\nDoc {i}: {type(e).__name__}: {e}")

        completed[i] = response
        progress_f.write(json.dumps({"idx": i, "response": response}) + "\n")
        progress_f.flush()

    progress_f.close()
    print(f"Completed {len(completed)} assignments")

    # Build final output
    responses = [completed.get(i, "Error") for i in range(len(docs))]
    df["prompted_docs"] = docs
    df["responses"] = responses
    df.to_json(ASSIGNMENT_OUTPUT, lines=True, orient="records")
    print(f"Wrote {ASSIGNMENT_OUTPUT}")


def run_correction():
    print("\n" + "=" * 60)
    print("STEP 2: Topic Correction")
    print("=" * 60)
    correct_topics(
        API, MODEL,
        ASSIGNMENT_OUTPUT,
        "prompt/correction.txt",
        TOPIC_FILE,
        CORRECTION_OUTPUT,
        verbose=True,
    )


def run_evaluation():
    print("\n" + "=" * 60)
    print("STEP 3: Evaluation")
    print("=" * 60)

    output_file = CORRECTION_OUTPUT
    if not os.path.exists(output_file):
        output_file = ASSIGNMENT_OUTPUT

    df = pd.read_json(output_file, lines=True)
    print(f"Loaded {len(df)} records from {output_file}")

    output_pattern = r"\[(?:\d+)\] ([^:]+): (?:.+)"
    parsed, errors = [], 0
    for resp in df["responses"]:
        match = re.findall(output_pattern, str(resp))
        if match:
            parsed.append(match[0].strip())
        else:
            parsed.append("UNKNOWN")
            errors += 1

    df["parsed_output"] = parsed
    print(f"Parsed {len(parsed)} predictions ({errors} parse errors)")

    valid = df[df["parsed_output"] != "UNKNOWN"]
    print(f"Evaluating on {len(valid)} valid predictions")

    purity, inverse_purity, harmonic_purity = calculate_purity(
        "label", "parsed_output", valid
    )
    harmonic, ari, nmi = calculate_metrics("label", "parsed_output", valid)

    print("\n" + "=" * 60)
    print("RESULTS: TopicGPT on 20newsgroups (FULL DATASET)")
    print("=" * 60)
    print(f"  Purity:          {purity:.4f}")
    print(f"  Inverse Purity:  {inverse_purity:.4f}")
    print(f"  Harmonic Purity: {harmonic_purity:.4f}")
    print(f"  ARI:             {ari:.4f}")
    print(f"  NMI:             {nmi:.4f}")
    print(f"  Total docs:      {len(df)}")
    print(f"  Valid docs:      {len(valid)}")
    print(f"  Parse errors:    {errors}")
    print("=" * 60)

    print("\n=== Predicted topic distribution ===")
    print(valid["parsed_output"].value_counts().to_string())

    print("\n=== Ground Truth -> Most Common Prediction ===")
    for label in sorted(valid["label"].unique()):
        subset = valid[valid["label"] == label]
        top_pred = subset["parsed_output"].value_counts().head(3)
        mapping = ", ".join([f"{k} ({v})" for k, v in top_pred.items()])
        print(f"  {label:30s} -> {mapping}")


if __name__ == "__main__":
    robust_assign()
    run_correction()
    run_evaluation()
