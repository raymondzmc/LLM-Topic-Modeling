#!/bin/bash
# Run retrieval baseline evaluations for all datasets x LLM models x baselines
#
# Baselines:
#   - bow: Bag-of-Words term frequency vectors
#   - llm_targets: LLM next-token probability distributions
#
# Datasets: 20_newsgroups, tweet_topic, stackoverflow
# LLM Models: ERNIE-4.5-0.3B-PT, Llama-3.1-8B-Instruct, Llama-3.2-1B-Instruct

DATASETS=(
    "20_newsgroups"
    "tweet_topic"
    "stackoverflow"
)

MODELS=(
    "ERNIE-4.5-0.3B-PT"
    "Llama-3.1-8B-Instruct"
    "Llama-3.2-1B-Instruct"
)

BASELINES=(
    "bow"
    "llm_targets"
)

echo "=========================================="
echo "Running Retrieval Baseline Evaluations"
echo "=========================================="

for dataset in "${DATASETS[@]}"
do
    for model in "${MODELS[@]}"
    do
        DATA_PATH="raymondzmc/${dataset}_${model}_vocab_2000_last"
        
        for baseline in "${BASELINES[@]}"
        do
            echo ""
            echo "============================================================"
            echo "Dataset: $dataset | Model: $model | Baseline: $baseline"
            echo "============================================================"
            
            python evaluate_retrieval_baselines.py \
                --data_path "$DATA_PATH" \
                --baseline "$baseline" \
                --wandb_project "${dataset}"
        done
    done
done

echo ""
echo "=========================================="
echo "All baseline evaluations complete!"
echo "=========================================="

