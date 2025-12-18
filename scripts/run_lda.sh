#!/bin/bash
# Run LDA baseline

datasets=(
    # "raymondzmc/20_newsgroups_Llama-3.1-8B-Instruct_vocab_2000_last"
    "raymondzmc/tweet_topic_Llama-3.1-8B-Instruct_vocab_2000_last"
    "raymondzmc/stackoverflow_Llama-3.1-8B-Instruct_vocab_2000_last"
)

for data in "${datasets[@]}"
do
    for K in 25 50 75 100
    do
        echo "Running LDA on $data with K=$K"
        python run_topic_model.py \
            --model lda \
            --data_path "$data" \
            --num_topics $K
    done
done
