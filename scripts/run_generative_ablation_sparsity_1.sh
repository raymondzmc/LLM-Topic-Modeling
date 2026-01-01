#!/bin/bash
# Run Generative Topic Model with sparsity_ratio ablation (Part 1)
# Model: Llama-3.2-1B-Instruct
# Sparsity ratios: 0.6, 0.5, 0.4

model="Llama-3.2-1B-Instruct"

datasets=(
    "20_newsgroups"
    "tweet_topic"
    "stackoverflow"
)

sparsity_ratios=(
    0.6
    0.5
    0.4
)

for dataset in "${datasets[@]}"
do
    DATA_PATH="raymondzmc/${dataset}_${model}_vocab_2000_last"
    
    for K in 25 50 75 100
    do
        for sparsity in "${sparsity_ratios[@]}"
        do
            echo "Running Generative TM on $DATA_PATH with K=$K, sparsity_ratio=$sparsity"
            python run_topic_model.py \
                --model generative \
                --data_path "$DATA_PATH" \
                --num_topics $K \
                --sparsity_ratio $sparsity
        done
    done
done
