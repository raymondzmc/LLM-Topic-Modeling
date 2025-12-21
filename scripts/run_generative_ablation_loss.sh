#!/bin/bash
# Run Generative Topic Model Ablation Experiments
# Tests two ablation conditions:
#   1. use_bow_target: Use BoW as target instead of LLM predictions
#   2. embedding_model: Use a different embedding model (gte-large-en-v1.5)
# Iterates over 3 datasets x 3 base models

datasets=(
    "20_newsgroups"
    "tweet_topic"
    "stackoverflow"
)

models=(
    "Llama-3.1-8B-Instruct"
    "ERNIE-4.5-0.3B-PT"
    "Llama-3.2-1B-Instruct"
)

for dataset in "${datasets[@]}"
do
    for model in "${models[@]}"
    do
        # Construct full dataset path/name
        DATA_PATH="raymondzmc/${dataset}_${model}_vocab_2000_last"
        
        for K in 25 50 75 100
        do
            echo "Running Generative TM (CE loss) on $DATA_PATH with K=$K"
            python run_topic_model.py \
                --model generative \
                --data_path "$DATA_PATH" \
                --num_topics $K \
                --loss_type "CE" \
                --temperature 1.0
        done
    done
done
