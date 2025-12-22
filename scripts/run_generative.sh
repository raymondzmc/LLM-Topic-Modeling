#!/bin/bash
# Run Generative Topic Model
# Iterates over 3 datasets x 3 base models

models=(
    "ERNIE-4.5-0.3B-PT"
    "Llama-3.1-8B-Instruct"
    "Llama-3.2-1B-Instruct"
)

datasets=(
    "20_newsgroups"
    "tweet_topic"
    "stackoverflow"
)



for model in "${models[@]}"
do
    for dataset in "${datasets[@]}"
    do
        # Construct full dataset path/name
        DATA_PATH="raymondzmc/${dataset}_${model}_vocab_2000_last"
        
        for K in 25 50 75 100
        do
            echo "Running Generative TM on $DATA_PATH with K=$K"
            python run_topic_model.py \
                --model generative \
                --data_path "$DATA_PATH" \
                --num_topics $K
        done
    done
done
