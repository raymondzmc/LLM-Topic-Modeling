#!/bin/bash
# Comprehensive test script for W&B artifact integration
# Tests training, artifact upload, and re-evaluation for all models and datasets

# Configuration
NUM_EPOCHS=5
NUM_SEEDS=1
NUM_TOPICS=10
BATCH_SIZE=64
WANDB_PROJECT="test-wandb-integration"

# Datasets
DATASETS=(
    "raymondzmc/stackoverflow_Llama-3.1-8B-Instruct_vocab_2000_last"
    "raymondzmc/tweet_topic_ERNIE-4.5-0.3B-PT_vocab_2000_last"
)

# Models to test
# Baseline models that don't need LLM data
BASELINE_MODELS=("lda" "prodlda" "zeroshot" "combined" "etm" "bertopic" "fastopic")

# LLM-based model
LLM_MODELS=("generative")

# Track results
RESULTS_FILE="test_results_$(date +%Y%m%d_%H%M%S).txt"
PASSED=0
FAILED=0

log() {
    echo "[$(date +%H:%M:%S)] $1" | tee -a "$RESULTS_FILE"
}

test_model() {
    local dataset=$1
    local model=$2
    local test_name="$model on $(basename $dataset)"
    
    log "Testing: $test_name"
    
    if python run_topic_model.py \
        --data_path "$dataset" \
        --model "$model" \
        --num_topics $NUM_TOPICS \
        --num_epochs $NUM_EPOCHS \
        --num_seeds $NUM_SEEDS \
        --batch_size $BATCH_SIZE \
        --wandb_project "$WANDB_PROJECT" 2>&1 | tee -a "$RESULTS_FILE"; then
        log "✓ PASSED: $test_name"
        ((PASSED++))
        return 0
    else
        log "✗ FAILED: $test_name"
        ((FAILED++))
        return 1
    fi
}

test_reevaluate() {
    local run_name=$1
    local test_name="Re-evaluate $run_name"
    
    log "Testing: $test_name"
    
    if python run_topic_model.py \
        --load_run_id_or_name "$run_name" \
        --wandb_project "$WANDB_PROJECT" 2>&1 | tee -a "$RESULTS_FILE"; then
        log "✓ PASSED: $test_name"
        ((PASSED++))
        return 0
    else
        log "✗ FAILED: $test_name"
        ((FAILED++))
        return 1
    fi
}

# ============================================================
# Main Test Suite
# ============================================================

echo "============================================================"
echo "W&B Integration Test Suite"
echo "============================================================"
echo "Project: $WANDB_PROJECT"
echo "Epochs: $NUM_EPOCHS, Seeds: $NUM_SEEDS, Topics: $NUM_TOPICS"
echo "Results file: $RESULTS_FILE"
echo "============================================================"
echo ""

# Test 1: Quick sanity check with one baseline model on one dataset
log "=== Phase 1: Sanity Check ==="
SANITY_DATASET="${DATASETS[1]}"  # tweet_topic_ERNIE (smaller)
SANITY_MODEL="prodlda"

log "Running sanity check: $SANITY_MODEL on $(basename $SANITY_DATASET)"
test_model "$SANITY_DATASET" "$SANITY_MODEL"

# Test 2: Test re-evaluation on the sanity check run
log ""
log "=== Phase 2: Re-evaluation Test ==="
REEVALUATE_RUN="${SANITY_MODEL}_K${NUM_TOPICS}"
log "Testing re-evaluation of: $REEVALUATE_RUN"
test_reevaluate "$REEVALUATE_RUN"

# Test 3: Test all baseline models on first dataset
log ""
log "=== Phase 3: All Baseline Models ==="
TEST_DATASET="${DATASETS[1]}"  # tweet_topic

for model in "${BASELINE_MODELS[@]}"; do
    test_model "$TEST_DATASET" "$model"
done

# Test 4: Test generative model (requires LLM-processed data)
log ""
log "=== Phase 4: Generative Model ==="
for model in "${LLM_MODELS[@]}"; do
    test_model "$TEST_DATASET" "$model"
done

# Test 5: Test across different datasets with one model
log ""
log "=== Phase 5: Cross-Dataset Test ==="
CROSS_MODEL="prodlda"

for dataset in "${DATASETS[@]}"; do
    test_model "$dataset" "$CROSS_MODEL"
done

# ============================================================
# Summary
# ============================================================

echo ""
echo "============================================================"
echo "Test Summary"
echo "============================================================"
log "Passed: $PASSED"
log "Failed: $FAILED"
log "Total: $((PASSED + FAILED))"
echo "============================================================"

if [ $FAILED -eq 0 ]; then
    log "All tests passed! ✓"
    exit 0
else
    log "Some tests failed! ✗"
    exit 1
fi

