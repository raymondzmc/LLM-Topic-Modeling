#!/bin/bash
# Run LDA baseline

DATA_PATH="data/processed_data/20_newsgroups_Llama-3.1-8B-Instruct_vocab_2000_last"

for K in 25 50 75 100
do
    echo "Running LDA with K=$K"
    python run_topic_model.py \
        --model lda \
        --data_path "$DATA_PATH" \
        --num_topics $K
done

