#!/bin/bash
# Train/Test Evaluation Script for StackOverflow Dataset
# Trains generative topic model on train set, evaluates clustering metrics on test set
# Supports ERNIE-4.5-0.3B-PT, Llama-3.2-1B-Instruct, and Llama-3.1-8B-Instruct

# ============================================================================
# Step 1: Process train datasets
# ============================================================================

echo "=========================================="
echo "Processing training datasets..."
echo "=========================================="

# stackoverflow_train.tsv with ERNIE-4.5-0.3B-PT
python process_dataset.py \
    --dataset data/raw_data/stackoverflow_train.tsv \
    --content_key text \
    --label_key label \
    --vocab_size 2000 \
    --model_name baidu/ERNIE-4.5-0.3B-PT \
    --batch_size 64 \
    --embedding_method last \
    --save_name stackoverflow_train_ERNIE-4.5-0.3B-PT_vocab_2000_last \
    --no_upload

# stackoverflow_train.tsv with Llama-3.2-1B-Instruct
python process_dataset.py \
    --dataset data/raw_data/stackoverflow_train.tsv \
    --content_key text \
    --label_key label \
    --vocab_size 2000 \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --batch_size 64 \
    --embedding_method last \
    --save_name stackoverflow_train_Llama-3.2-1B-Instruct_vocab_2000_last \
    --no_upload

# stackoverflow_train.tsv with Llama-3.1-8B-Instruct
python process_dataset.py \
    --dataset data/raw_data/stackoverflow_train.tsv \
    --content_key text \
    --label_key label \
    --vocab_size 2000 \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --batch_size 1 \
    --embedding_method last \
    --save_name stackoverflow_train_Llama-3.1-8B-Instruct_vocab_2000_last \
    --no_upload

# ============================================================================
# Step 2: Process test datasets (reusing vocab from train)
# ============================================================================

echo "=========================================="
echo "Processing test datasets..."
echo "=========================================="

# stackoverflow_test.tsv with ERNIE-4.5-0.3B-PT (using train vocab)
python process_dataset.py \
    --dataset data/raw_data/stackoverflow_test.tsv \
    --vocab_path data/processed_data/stackoverflow_train_ERNIE-4.5-0.3B-PT_vocab_2000_last/vocab.json \
    --content_key text \
    --label_key label \
    --model_name baidu/ERNIE-4.5-0.3B-PT \
    --batch_size 64 \
    --embedding_method last \
    --save_name stackoverflow_test_ERNIE-4.5-0.3B-PT_vocab_2000_last \
    --no_upload

# stackoverflow_test.tsv with Llama-3.2-1B-Instruct (using train vocab)
python process_dataset.py \
    --dataset data/raw_data/stackoverflow_test.tsv \
    --vocab_path data/processed_data/stackoverflow_train_Llama-3.2-1B-Instruct_vocab_2000_last/vocab.json \
    --content_key text \
    --label_key label \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --batch_size 64 \
    --embedding_method last \
    --save_name stackoverflow_test_Llama-3.2-1B-Instruct_vocab_2000_last \
    --no_upload

# stackoverflow_test.tsv with Llama-3.1-8B-Instruct (using train vocab)
python process_dataset.py \
    --dataset data/raw_data/stackoverflow_test.tsv \
    --vocab_path data/processed_data/stackoverflow_train_Llama-3.1-8B-Instruct_vocab_2000_last/vocab.json \
    --content_key text \
    --label_key label \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --batch_size 1 \
    --embedding_method last \
    --save_name stackoverflow_test_Llama-3.1-8B-Instruct_vocab_2000_last \
    --no_upload

# ============================================================================
# Step 3: Run train/test evaluation for each model
# ============================================================================

echo "=========================================="
echo "Running train/test evaluations..."
echo "=========================================="

# Create results directory
mkdir -p results/train_test_eval

# Models to evaluate
models=(
    "ERNIE-4.5-0.3B-PT"
    "Llama-3.2-1B-Instruct"
    "Llama-3.1-8B-Instruct"
)

for model in "${models[@]}"
do
    echo ""
    echo "=========================================="
    echo "Evaluating model: ${model}"
    echo "=========================================="
    
    python evaluate_train_test.py \
        --train_data_path "data/processed_data/stackoverflow_train_${model}_vocab_2000_last" \
        --test_data_path "data/processed_data/stackoverflow_test_${model}_vocab_2000_last" \
        --num_topics 22 \
        --num_seeds 5 \
        --output_path "results/train_test_eval/${model}_results.json"
done

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "Results saved to: results/train_test_eval/"
echo "=========================================="
