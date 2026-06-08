#!/bin/bash
# Run generative ProdLDA with top-k sparse targets
# ERNIE-4.5-0.3B on TweetTopic and StackOverflow
# TopK in {10, 20, 50, 100, 500}, 5 seeds each
# (topk=2000 is equivalent to sparsity_ratio=1.0, already have those results)

PYTHON="/home/toolkit/.conda/envs/llm-topics/bin/python"
HF_USER="raymondzmc"
LLM="ERNIE-4.5-0.3B-PT"

for dataset in "tweet_topic"
do
    DATA_NAME="${dataset}_${LLM}_vocab_2000_last"
    if [ -d "data/processed_data/${DATA_NAME}" ]; then
        DATA_PATH="data/processed_data/${DATA_NAME}"
    else
        DATA_PATH="${HF_USER}/${DATA_NAME}"
    fi

    for TOPK in 1000
    do
        for K in 50
        do
            echo "=== generative topk=${TOPK}: ${LLM}, ${dataset}, K=${K} ==="
            $PYTHON run_topic_model.py \
                --model generative \
                --data_path "$DATA_PATH" \
                --num_topics $K \
                --topk $TOPK \
                --num_seeds 5
        done
    done
done

echo "All generative top-k runs complete."
