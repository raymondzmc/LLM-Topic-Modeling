#!/bin/bash
# Run all remaining generative_fastopic experiments identified by audit_runs.py.
#
# As of latest audit: 275/276 complete; only 1 job left.
# (Previously failed due to external GPU contention, not OOM in our code —
#  three other processes were occupying ~78 GiB on GPU 0.)

PYTHON="/home/toolkit/.conda/envs/llm-topics/bin/python"
HF_USER="raymondzmc"

# FASTopic K=100 OOMs were caused by Sinkhorn autograd-graph accumulation;
# fixed in models/fastopic/_ETP.py by lowering OT_max_iter 5000 -> 1000
# (verified on K=75 to produce equivalent metrics, ±0.01 across CV/LLM/Purity).
# expandable_segments kept as a small safety margin against fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
    CUDA_VISIBLE_DEVICES=0 $PYTHON run_topic_model.py \
        --model generative_fastopic \
        --data_path "$DATA_PATH" \
        --num_topics $K
}

# --- Last remaining: Qwen3.5-0.8B × 20_newsgroups × K=100 ---
run_job "Qwen3.5-0.8B"  "20_newsgroups"  100

echo "All remaining experiments complete."
