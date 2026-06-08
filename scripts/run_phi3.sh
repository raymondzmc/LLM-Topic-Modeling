#!/bin/bash
# Full pipeline for Phi-3-mini-128k-instruct on GPU 0
# Step 1: Process all 3 datasets (batch_size=4 to avoid OOM on long docs)
# Step 2: Train Generative TM (K=25,50,75,100 x 3 datasets)

export CUDA_VISIBLE_DEVICES=0
PYTHON="/home/toolkit/.conda/envs/llm-topics/bin/python"
DATA_DIR="data/processed_data"

echo "=== [GPU 0] Phi-3-mini-128k-instruct ==="

# # --- Step 1: Data ---

# echo "[GPU 0] Processing tweet_topic with Phi-3..."
# $PYTHON process_dataset.py \
#     --dataset data/raw_data/tweet_topic.tsv \
#     --content_key text \
#     --label_key label \
#     --vocab_size 2000 \
#     --model_name microsoft/Phi-3-mini-128k-instruct \
#     --batch_size 4 \
#     --embedding_method last \
#     --save_name tweet_topic_Phi-3-mini-128k-instruct_vocab_2000_last

# echo "[GPU 0] Processing stackoverflow with Phi-3..."
# $PYTHON process_dataset.py \
#     --dataset data/raw_data/stackoverflow.tsv \
#     --content_key text \
#     --label_key label \
#     --vocab_size 2000 \
#     --model_name microsoft/Phi-3-mini-128k-instruct \
#     --batch_size 4 \
#     --embedding_method last \
#     --save_name stackoverflow_Phi-3-mini-128k-instruct_vocab_2000_last

# echo "[GPU 0] Processing 20_newsgroups with Phi-3..."
# $PYTHON process_dataset.py \
#     --dataset SetFit/20_newsgroups \
#     --content_key text \
#     --label_key label \
#     --split all \
#     --vocab_size 2000 \
#     --model_name microsoft/Phi-3-mini-128k-instruct \
#     --batch_size 4 \
#     --embedding_method last \
#     --save_name 20_newsgroups_Phi-3-mini-128k-instruct_vocab_2000_last

# echo "[GPU 0] All Phi-3 data ready."

# # --- Step 2: Train ---

datasets=(
    # "tweet_topic"
    # "stackoverflow"
    "20_newsgroups"
)

for dataset in "${datasets[@]}"
do
    DATA_PATH="${DATA_DIR}/${dataset}_Phi-3-mini-128k-instruct_vocab_2000_last"

    for K in 25 50 75 100
    do
        echo "[GPU 0] Generative TM: Phi-3, dataset=${dataset}, K=${K}"
        $PYTHON run_topic_model.py \
            --model generative \
            --data_path "$DATA_PATH" \
            --num_topics $K
    done
done

echo "[GPU 0] All Phi-3 runs complete."
