#!/bin/bash
# Run Generative Topic Model with temperature ablation (Part 3)
# Model: ERNIE-4.5-0.3B-PT
# Temperatures: 1, 2, 3

model="ERNIE-4.5-0.3B-PT"

datasets=(
    "20_newsgroups"
    "tweet_topic"
    "stackoverflow"
)

temperatures=(
    1
    2
    3
)

for dataset in "${datasets[@]}"
do
    DATA_PATH="raymondzmc/${dataset}_${model}_vocab_2000_last"
    
    for K in 25 50 75 100
    do
        for temp in "${temperatures[@]}"
        do
            echo "Running Generative TM on $DATA_PATH with K=$K, temperature=$temp"
            python run_topic_model.py \
                --model generative \
                --data_path "$DATA_PATH" \
                --num_topics $K \
                --temperature $temp
        done
    done
done

