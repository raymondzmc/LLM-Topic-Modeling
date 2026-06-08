#!/bin/bash
# Train generative_ecrtm — only unfinished jobs
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
    echo "=== generative_ecrtm: ${llm}, ${dataset}, K=${K} ==="
    $PYTHON run_topic_model.py \
        --model generative_ecrtm \
        --data_path "$DATA_PATH" \
        --num_topics $K
}

# --- DONE ---
# ERNIE-4.5-0.3B-PT: all datasets, all K — COMPLETE
# Llama-3.1-8B-Instruct: all datasets, all K — COMPLETE
# Llama-3.2-1B-Instruct: all datasets, all K — COMPLETE
# Phi-3-mini-128k-instruct: tweet_topic (K=25,50,75,100) — COMPLETE
# Phi-3-mini-128k-instruct: stackoverflow (K=25,50,75,100) — COMPLETE
# Phi-3-mini-128k-instruct: 20_newsgroups (K=25,50) — COMPLETE

# --- Phi-3-mini: 20_newsgroups K=75 (was killed), K=100 (not started) ---
run_job "Phi-3-mini-128k-instruct" "20_newsgroups" 75
run_job "Phi-3-mini-128k-instruct" "20_newsgroups" 100

# --- Qwen3.5-0.8B: all datasets, all K ---
run_job "Qwen3.5-0.8B" "tweet_topic" 25
run_job "Qwen3.5-0.8B" "tweet_topic" 50
run_job "Qwen3.5-0.8B" "tweet_topic" 75
run_job "Qwen3.5-0.8B" "tweet_topic" 100
run_job "Qwen3.5-0.8B" "stackoverflow" 25
run_job "Qwen3.5-0.8B" "stackoverflow" 50
run_job "Qwen3.5-0.8B" "stackoverflow" 75
run_job "Qwen3.5-0.8B" "stackoverflow" 100
run_job "Qwen3.5-0.8B" "20_newsgroups" 25
run_job "Qwen3.5-0.8B" "20_newsgroups" 50
run_job "Qwen3.5-0.8B" "20_newsgroups" 75
run_job "Qwen3.5-0.8B" "20_newsgroups" 100

echo "All remaining generative_ecrtm runs complete."
