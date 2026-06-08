"""
Run TopicGPT pipeline on 20newsgroups dataset (full 11,314 docs).

Usage: python scripts/run_topicgpt_20newsgroups.py
"""

import os
import sys
import yaml
import re
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

TOPICGPT_DIR = os.path.join(PROJECT_ROOT, "topicGPT")
sys.path.insert(0, TOPICGPT_DIR)
os.chdir(TOPICGPT_DIR)

from topicgpt_python import (
    generate_topic_lvl1,
    refine_topics,
    assign_topics,
    correct_topics,
)
from topicgpt_python.utils import calculate_purity, calculate_metrics

API = "openai"
GEN_MODEL = "gpt-4o-mini"
ASSIGN_MODEL = "gpt-4o-mini"


def load_config():
    with open("config_20newsgroups.yml", "r") as f:
        return yaml.safe_load(f)


def run_generation(config):
    print("\n" + "=" * 60)
    print("STEP 1: Topic Generation (Level 1)")
    print("=" * 60)
    generate_topic_lvl1(
        API,
        GEN_MODEL,
        config["data_sample"],
        config["generation"]["prompt"],
        config["generation"]["seed"],
        config["generation"]["output"],
        config["generation"]["topic_output"],
        verbose=config["verbose"],
    )


def run_refinement(config):
    print("\n" + "=" * 60)
    print("STEP 2: Topic Refinement")
    print("=" * 60)
    if not config.get("refining_topics", False):
        print("Skipping refinement (disabled in config).")
        return
    refine_topics(
        API,
        GEN_MODEL,
        config["refinement"]["prompt"],
        config["generation"]["output"],
        config["generation"]["topic_output"],
        config["refinement"]["topic_output"],
        config["refinement"]["output"],
        verbose=config["verbose"],
        remove=config["refinement"]["remove"],
        mapping_file=config["refinement"]["mapping_file"],
    )


def run_assignment(config):
    print("\n" + "=" * 60)
    print("STEP 3: Topic Assignment (full 11,314 docs)")
    print("=" * 60)
    topic_file = config["generation"]["topic_output"]
    if config.get("refining_topics") and os.path.exists(
        config["refinement"]["topic_output"]
    ):
        topic_file = config["refinement"]["topic_output"]
        print(f"Using refined topics: {topic_file}")
    assign_topics(
        API,
        ASSIGN_MODEL,
        config["data_full"],
        config["assignment"]["prompt"],
        config["assignment"]["output"],
        topic_file,
        verbose=config["verbose"],
    )


def run_correction(config):
    print("\n" + "=" * 60)
    print("STEP 4: Topic Correction")
    print("=" * 60)
    topic_file = config["generation"]["topic_output"]
    if config.get("refining_topics") and os.path.exists(
        config["refinement"]["topic_output"]
    ):
        topic_file = config["refinement"]["topic_output"]
    correct_topics(
        API,
        ASSIGN_MODEL,
        config["assignment"]["output"],
        config["correction"]["prompt"],
        topic_file,
        config["correction"]["output"],
        verbose=config["verbose"],
    )


def run_evaluation(config):
    print("\n" + "=" * 60)
    print("STEP 5: Evaluation")
    print("=" * 60)
    output_file = config["correction"]["output"]
    if not os.path.exists(output_file):
        output_file = config["assignment"]["output"]

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
    print("RESULTS: TopicGPT on 20newsgroups")
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
    config = load_config()
    run_generation(config)
    run_refinement(config)
    run_assignment(config)
    run_correction(config)
    run_evaluation(config)
