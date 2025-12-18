#!/bin/bash
# Script to run generative topic model and upload results to wandb

# Default parameters
DATA_PATH="${DATA_PATH:raymondzmc/tweet_topic_ERNIE-4.5-0.3B-PT_vocab_2000_last}"
NUM_TOPICS="${NUM_TOPICS:25}"
NUM_SEEDS="${NUM_SEEDS:5}"
NUM_EPOCHS="${NUM_EPOCHS:100}"
TOP_WORDS="${TOP_WORDS:10}"
BATCH_SIZE="${BATCH_SIZE:64}"
LR="${LR:0.002}"

echo "Running generative topic model..."
echo "  Data: ${DATA_PATH}"
echo "  Topics: ${NUM_TOPICS}"
echo "  Seeds: ${NUM_SEEDS}"
echo "  Epochs: ${NUM_EPOCHS}"

python run_topic_model.py \
    --data_path "${DATA_PATH}" \
    --model generative \
    --num_topics ${NUM_TOPICS} \
    --num_seeds ${NUM_SEEDS} \
    --num_epochs ${NUM_EPOCHS} \
    --top_words ${TOP_WORDS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR}

echo "Done! Results uploaded to wandb."

