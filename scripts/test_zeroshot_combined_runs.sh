#!/bin/bash

# Initialize conda for shell interaction
eval "$(conda shell.bash hook)"
conda activate llm-topics

# Dataset path
# Using the locally available processed dataset
DATA_PATH="$(pwd)/data/processed_data/20_newsgroups_ERNIE-4.5-0.3B-PT_vocab_2000_last"

# Test ZeroShot
echo "==========================================="
echo "Testing ZeroShotTM (K=5, Epochs=5)..."
echo "==========================================="
python run_topic_model.py \
    --model zeroshot \
    --data_path "$DATA_PATH" \
    --num_topics 5 \
    --num_epochs 5 \
    --num_seeds 1 \
    --wandb_offline

# Test Combined
echo "==========================================="
echo "Testing CombinedTM (K=5, Epochs=5)..."
echo "==========================================="
python run_topic_model.py \
    --model combined \
    --data_path "$DATA_PATH" \
    --num_topics 5 \
    --num_epochs 5 \
    --num_seeds 1 \
    --wandb_offline

echo "==========================================="
echo "Tests completed!"
echo "==========================================="

