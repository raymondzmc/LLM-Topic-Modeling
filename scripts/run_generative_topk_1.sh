#!/bin/bash
# Run generative ProdLDA with top-k sparse targets
# ERNIE-4.5-0.3B on StackOverflow only, GPU 1
# TopK in {10, 20, 50, 100, 500, 1000}, K=50, 5 seeds each

PYTHON="/home/toolkit/.conda/envs/llm-topics/bin/python"
HF_USER="raymondzmc"
LLM="ERNIE-4.5-0.3B-PT"
dataset="stackoverflow"
DATA_NAME="${dataset}_${LLM}_vocab_2000_last"
if [ -d "data/processed_data/${DATA_NAME}" ]; then
    DATA_PATH="data/processed_data/${DATA_NAME}"
else
    DATA_PATH="${HF_USER}/${DATA_NAME}"
fi

for TOPK in 10 20 50 100 500 1000
do
    echo "=== generative topk=${TOPK}: ${LLM}, ${dataset}, K=50 ==="
    CUDA_VISIBLE_DEVICES=1 $PYTHON run_topic_model.py \
        --model generative \
        --data_path "$DATA_PATH" \
        --num_topics 50 \
        --topk $TOPK \
        --num_seeds 5
done

echo "All StackOverflow top-k runs complete."