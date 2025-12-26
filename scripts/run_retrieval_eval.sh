#!/bin/bash
# Run retrieval evaluation for all finished runs across all projects

# Set wandb cache directories to avoid permission issues
export WANDB_CACHE_DIR=/tmp/wandb_cache
export WANDB_DIR=/tmp/wandb
export WANDB_DATA_DIR=/tmp/wandb_data

# Projects to evaluate
PROJECTS=("20_newsgroups" "stackoverflow" "tweet_topic")

for project in "${PROJECTS[@]}"; do
    echo ""
    echo "========================================"
    echo "Evaluating project: $project"
    echo "========================================"
    echo ""
    
    python evaluate_retrieval.py --wandb_project "$project" --all
done

echo ""
echo "========================================"
echo "All projects evaluated successfully!"
echo "========================================"

