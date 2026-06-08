#!/bin/bash
# Full pipeline for Qwen3.5-0.8B on GPU 1
# Step 1: Process data
# Step 2: Train Generative TM (K=25,50,75,100 x 3 datasets)

export CUDA_VISIBLE_DEVICES=1
PYTHON="/home/toolkit/.conda/envs/llm-topics/bin/python"
DATA_DIR="data/processed_data"

echo "=== [GPU 1] Qwen3.5-0.8B ==="

# --- Step 1: Data ---

echo "[GPU 1] Processing tweet_topic with Qwen3.5..."
$PYTHON process_dataset.py \
    --dataset data/raw_data/tweet_topic.tsv \
    --content_key text \
    --label_key label \
    --vocab_size 2000 \
    --model_name Qwen/Qwen3.5-0.8B \
    --batch_size 64 \
    --embedding_method last \
    --save_name tweet_topic_Qwen3.5-0.8B_vocab_2000_last

echo "[GPU 1] Processing stackoverflow with Qwen3.5..."
$PYTHON process_dataset.py \
    --dataset data/raw_data/stackoverflow.tsv \
    --content_key text \
    --label_key label \
    --vocab_size 2000 \
    --model_name Qwen/Qwen3.5-0.8B \
    --batch_size 64 \
    --embedding_method last \
    --save_name stackoverflow_Qwen3.5-0.8B_vocab_2000_last

echo "[GPU 1] Processing 20_newsgroups with Qwen3.5..."
$PYTHON process_dataset.py \
    --dataset SetFit/20_newsgroups \
    --content_key text \
    --label_key label \
    --split all \
    --vocab_size 2000 \
    --model_name Qwen/Qwen3.5-0.8B \
    --batch_size 64 \
    --embedding_method last \
    --save_name 20_newsgroups_Qwen3.5-0.8B_vocab_2000_last

echo "[GPU 1] All Qwen3.5 data ready."

# --- Step 2: Train ---

datasets=(
    "tweet_topic"
    "stackoverflow"
    "20_newsgroups"
)

for dataset in "${datasets[@]}"
do
    DATA_PATH="${DATA_DIR}/${dataset}_Qwen3.5-0.8B_vocab_2000_last"

    for K in 25 50 75 100
    do
        echo "[GPU 1] Generative TM: Qwen3.5, dataset=${dataset}, K=${K}"
        $PYTHON run_topic_model.py \
            --model generative \
            --data_path "$DATA_PATH" \
            --num_topics $K
    done
done

echo "[GPU 1] All Qwen3.5 runs complete."
