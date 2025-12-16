#!/usr/bin/env bash
# Test script to process small subsets of datasets with different models

export PYTHONPATH=.
SUBSET=5
VOCAB_SIZE=50

# Array of models to test
MODELS=(
    "baidu/ERNIE-4.5-0.3B-PT"
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
)

# Test 1: stackoverflow.tsv (local TSV file)
echo "========================================="
echo "Testing stackoverflow.tsv"
echo "========================================="

for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT=$(echo $MODEL | sed 's/.*\///')
    SAVE_NAME="test_stackoverflow_${MODEL_SHORT}_subset${SUBSET}"
    
    echo "Processing with model: $MODEL"
    
    python data/process_dataset.py \
        --dataset data/raw_data/stackoverflow.tsv \
        --content_key text \
        --label_key label \
        --model_name "$MODEL" \
        --vocab_size $VOCAB_SIZE \
        --subset $SUBSET \
        --save_name "$SAVE_NAME" \
        --batch_size 2 \
        --hf_repo_name "raymondzmc/test_${SAVE_NAME}" \
        --hf_private
    
    if [ $? -eq 0 ]; then
        echo "✓ SUCCESS: $MODEL with stackoverflow.tsv"
    else
        echo "✗ FAILED: $MODEL with stackoverflow.tsv"
    fi
    echo ""
done

# Test 2: tweet_topic.tsv (local TSV file)
echo "========================================="
echo "Testing tweet_topic.tsv"
echo "========================================="

for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT=$(echo $MODEL | sed 's/.*\///')
    SAVE_NAME="test_tweet_topic_${MODEL_SHORT}_subset${SUBSET}"
    
    echo "Processing with model: $MODEL"
    
    python data/process_dataset.py \
        --dataset data/raw_data/tweet_topic.tsv \
        --content_key text \
        --label_key label_name \
        --model_name "$MODEL" \
        --vocab_size $VOCAB_SIZE \
        --subset $SUBSET \
        --save_name "$SAVE_NAME" \
        --batch_size 2 \
        --hf_repo_name "raymondzmc/test_${SAVE_NAME}" \
        --hf_private
    
    if [ $? -eq 0 ]; then
        echo "✓ SUCCESS: $MODEL with tweet_topic.tsv"
    else
        echo "✗ FAILED: $MODEL with tweet_topic.tsv"
    fi
    echo ""
done

# Test 3: SetFit/20_newsgroups (HuggingFace dataset)
echo "========================================="
echo "Testing SetFit/20_newsgroups"
echo "========================================="

for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT=$(echo $MODEL | sed 's/.*\///')
    SAVE_NAME="test_20newsgroups_${MODEL_SHORT}_subset${SUBSET}"
    
    echo "Processing with model: $MODEL"
    
    python data/process_dataset.py \
        --dataset SetFit/20_newsgroups \
        --content_key text \
        --label_key label \
        --model_name "$MODEL" \
        --vocab_size $VOCAB_SIZE \
        --subset $SUBSET \
        --save_name "$SAVE_NAME" \
        --batch_size 2 \
        --hf_repo_name "raymondzmc/test_${SAVE_NAME}" \
        --hf_private
    
    if [ $? -eq 0 ]; then
        echo "✓ SUCCESS: $MODEL with SetFit/20_newsgroups"
    else
        echo "✗ FAILED: $MODEL with SetFit/20_newsgroups"
    fi
    echo ""
done

echo "========================================="
echo "All tests completed!"
echo "========================================="
