#!/bin/bash
# Validate all processed datasets for Phi-3 and Qwen3.5 before training.
# Exits with code 1 if any dataset fails validation.
set -e

PYTHON="/home/toolkit/.conda/envs/llm-topics/bin/python"
DATA_DIR="data/processed_data"

DATASETS=(
    "${DATA_DIR}/tweet_topic_Phi-3-mini-128k-instruct_vocab_2000_last"
    "${DATA_DIR}/stackoverflow_Phi-3-mini-128k-instruct_vocab_2000_last"
    "${DATA_DIR}/20_newsgroups_Phi-3-mini-128k-instruct_vocab_2000_last"
    "${DATA_DIR}/tweet_topic_Qwen3.5-0.8B_vocab_2000_last"
    "${DATA_DIR}/stackoverflow_Qwen3.5-0.8B_vocab_2000_last"
    "${DATA_DIR}/20_newsgroups_Qwen3.5-0.8B_vocab_2000_last"
)

$PYTHON scripts/validate_processed_data.py "${DATASETS[@]}"
