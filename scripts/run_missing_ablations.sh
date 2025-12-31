#!/bin/bash
# Re-run missing Llama-3.2-1B-Instruct ablation experiments
#
# Missing runs:
# - stackoverflow: bow-target (all K), nll (all K)
# - tweet_topic: bow-target (all K), nll (K75, K100)

MODEL="Llama-3.2-1B-Instruct"

echo "=========================================="
echo "Re-running missing Llama-3.2-1B-Instruct ablation experiments"
echo "=========================================="

# ==========================================
# ABLATION 1: BoW Target (stackoverflow - all K values)
# ==========================================
echo ""
echo "=== BoW Target Ablation: stackoverflow ==="
for K in 25 50 75 100
do
    DATA_PATH="raymondzmc/stackoverflow_${MODEL}_vocab_2000_last"
    echo "Running Generative TM (BoW target) on stackoverflow with K=$K"
    python run_topic_model.py \
        --model generative \
        --data_path "$DATA_PATH" \
        --num_topics $K \
        --ablation_use_bow_target \
        --loss_type "CE" \
        --temperature 1.0
done

# ==========================================
# ABLATION 1: BoW Target (tweet_topic - all K values)
# ==========================================
echo ""
echo "=== BoW Target Ablation: tweet_topic ==="
for K in 25 50 75 100
do
    DATA_PATH="raymondzmc/tweet_topic_${MODEL}_vocab_2000_last"
    echo "Running Generative TM (BoW target) on tweet_topic with K=$K"
    python run_topic_model.py \
        --model generative \
        --data_path "$DATA_PATH" \
        --num_topics $K \
        --ablation_use_bow_target \
        --loss_type "CE" \
        --temperature 1.0
done

# ==========================================
# ABLATION 2: NLL Loss (stackoverflow - all K values)
# ==========================================
echo ""
echo "=== NLL Loss Ablation: stackoverflow ==="
for K in 25 50 75 100
do
    DATA_PATH="raymondzmc/stackoverflow_${MODEL}_vocab_2000_last"
    echo "Running Generative TM (NLL loss) on stackoverflow with K=$K"
    python run_topic_model.py \
        --model generative \
        --data_path "$DATA_PATH" \
        --num_topics $K \
        --loss_type "CE" \
        --temperature 1.0
done

# ==========================================
# ABLATION 2: NLL Loss (tweet_topic - K75, K100 only)
# ==========================================
echo ""
echo "=== NLL Loss Ablation: tweet_topic (K75, K100 only) ==="
for K in 75 100
do
    DATA_PATH="raymondzmc/tweet_topic_${MODEL}_vocab_2000_last"
    echo "Running Generative TM (NLL loss) on tweet_topic with K=$K"
    python run_topic_model.py \
        --model generative \
        --data_path "$DATA_PATH" \
        --num_topics $K \
        --loss_type "CE" \
        --temperature 1.0
done

echo ""
echo "=========================================="
echo "All missing ablation experiments completed!"
echo "=========================================="

