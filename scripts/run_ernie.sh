#!/bin/bash
# Run Generative Topic Model with ERNIE-4.5-0.3B-PT

model="ERNIE-4.5-0.3B-PT"

datasets=(
    "20_newsgroups"
    "tweet_topic"
    "stackoverflow"
)

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

