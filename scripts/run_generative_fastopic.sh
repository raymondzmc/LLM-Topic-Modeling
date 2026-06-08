#!/bin/bash
# Train generative_fastopic — only unfinished jobs
# Data is auto-downloaded from HuggingFace if not cached locally.

PYTHON="/home/toolkit/.conda/envs/llm-topics/bin/python"
HF_USER="raymondzmc"


run_job() {
    local llm="$1"
    local dataset="$2"
    local K="$3"
    DATA_NAME="${dataset}_${llm}_vocab_2000_last"
    if [ -d "data/processed_data/${DATA_NAME}" ]; then
        DATA_PATH="data/processed_data/${DATA_NAME}"
    else
        DATA_PATH="${HF_USER}/${DATA_NAME}"
    fi
    echo "=== generative_fastopic: ${llm}, ${dataset}, K=${K} ==="
    CUDA_VISIBLE_DEVICES=1 $PYTHON run_topic_model.py \
        --model generative_fastopic \
        --data_path "$DATA_PATH" \
        --num_topics $K
}

# --- DONE ---
# ERNIE-4.5-0.3B-PT: tweet_topic (K=25,50,75), stackoverflow (K=25,50,75,100), 20_newsgroups (K=25,50,75,100)
# Llama-3.1-8B-Instruct: tweet_topic (K=25,50,75), stackoverflow (K=25,50,75), 20_newsgroups (K=25,50,75)
# Llama-3.2-1B-Instruct: tweet_topic (K=25,50,75), stackoverflow (K=25,50,75,100), 20_newsgroups (K=25,50,75,100)
# Phi-3-mini-128k-instruct: all datasets, all K — COMPLETE
# Qwen3.5-0.8B: all datasets, all K — COMPLETE

# --- K=100 failures (all failed/killed, need retry) ---
run_job "ERNIE-4.5-0.3B-PT" "tweet_topic" 100
run_job "Llama-3.1-8B-Instruct" "tweet_topic" 100
run_job "Llama-3.1-8B-Instruct" "stackoverflow" 100
run_job "Llama-3.1-8B-Instruct" "20_newsgroups" 100
run_job "Llama-3.2-1B-Instruct" "tweet_topic" 100

echo "All remaining generative_fastopic runs complete."
