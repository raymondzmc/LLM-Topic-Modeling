#!/bin/bash
# Test script to verify ZeroShotTM and CombinedTM execution
# Runs 5 epochs to ensure completion without errors.

# Dataset to use (local)
# Using just the folder name since run_topic_model.py looks in data/processed_data/
DATASET="20_newsgroups_ERNIE-4.5-0.3B-PT_vocab_2000_last"
TOPICS=5
EPOCHS=5

echo "----------------------------------------------------------------"
echo "Starting test for ZeroShotTM"
echo "Dataset: $DATASET"
echo "Topics: $TOPICS, Epochs: $EPOCHS"
echo "----------------------------------------------------------------"

python run_topic_model.py \
    --model zeroshot \
    --data_path "$DATASET" \
    --num_topics $TOPICS \
    --num_epochs $EPOCHS \
    --num_seeds 1 \
    --wandb_offline

if [ $? -ne 0 ]; then
    echo "ERROR: ZeroShotTM failed to run."
    exit 1
fi
echo "ZeroShotTM passed."

echo "----------------------------------------------------------------"
echo "Starting test for CombinedTM"
echo "Dataset: $DATASET"
echo "Topics: $TOPICS, Epochs: $EPOCHS"
echo "----------------------------------------------------------------"

python run_topic_model.py \
    --model combined \
    --data_path "$DATASET" \
    --num_topics $TOPICS \
    --num_epochs $EPOCHS \
    --num_seeds 1 \
    --wandb_offline

if [ $? -ne 0 ]; then
    echo "ERROR: CombinedTM failed to run."
    exit 1
fi
echo "CombinedTM passed."

echo "----------------------------------------------------------------"
echo "All tests completed successfully."
echo "----------------------------------------------------------------"

