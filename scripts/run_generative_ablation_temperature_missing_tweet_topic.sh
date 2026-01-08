#!/bin/bash
# Run missing temperature ablation experiments for tweet_topic
# Model: ERNIE-4.5-0.3B-PT
# Missing temperatures: 0.5, 1.0, 2.0, 6.0, 7.0, 9.0

model="ERNIE-4.5-0.3B-PT"
dataset="tweet_topic"
DATA_PATH="raymondzmc/${dataset}_${model}_vocab_2000_last"

# Missing temperatures for tweet_topic
temperatures=(
    0.5
    1.0
    2.0
    6.0
    7.0
    9.0
)

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

