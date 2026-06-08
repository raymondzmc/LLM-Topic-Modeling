"""
Run TopicGPT assignment+correction+evaluation on tweet_topic.
Generation and refinement are already complete.

Usage: python scripts/run_topicgpt_tweet_assign.py
"""

import os
import sys
import re
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

TOPICGPT_DIR = os.path.join(PROJECT_ROOT, "topicGPT")
sys.path.insert(0, TOPICGPT_DIR)
os.chdir(TOPICGPT_DIR)

from topicgpt_python import assign_topics, correct_topics
from topicgpt_python.utils import calculate_purity, calculate_metrics

API = "openai"
MODEL = "gpt-4o-mini"

DATA_FILE = "data/input/tweet_topic_eval.jsonl"
TOPIC_FILE = "data/output/tweet_topic/refinement.md"
ASSIGNMENT_OUTPUT = "data/output/tweet_topic/assignment.jsonl"
CORRECTION_OUTPUT = "data/output/tweet_topic/assignment_corrected.jsonl"


def run_assignment():
    print("\n" + "=" * 60)
    print("STEP 1: Topic Assignment (2000 doc sample)")
    print("=" * 60)
    assign_topics(
        API,
        MODEL,
        DATA_FILE,
        "prompt/assignment.txt",
        ASSIGNMENT_OUTPUT,
        TOPIC_FILE,
        verbose=True,
    )


def run_correction():
    print("\n" + "=" * 60)
    print("STEP 2: Topic Correction")
    print("=" * 60)
    correct_topics(
        API,
        MODEL,
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
    parsed = []
    errors = 0
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
    print("RESULTS: TopicGPT on tweet_topic")
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


if __name__ == "__main__":
    run_assignment()
    run_correction()
    run_evaluation()
