#!/bin/bash
# Run the embedding overhead benchmark
#
# Usage:
#   ./analysis/run_benchmark.sh              # Run full benchmark
#   ./analysis/run_benchmark.sh --quick      # Quick benchmark (100 docs)
#   ./analysis/run_benchmark.sh --embedding  # Only embedding models
#   ./analysis/run_benchmark.sh --llm        # Only LLM models

set -e

cd "$(dirname "$0")/.."

# Default settings
NUM_DOCS=1000
BATCH_SIZE_EMBEDDING=32
BATCH_SIZE_LLM=8
OUTPUT="analysis/benchmark_results.json"

# Parse arguments
SKIP_LLM=""
SKIP_EMBEDDING=""

for arg in "$@"; do
    case $arg in
        --quick)
            NUM_DOCS=100
            ;;
        --embedding)
            SKIP_LLM="--skip_llm"
            ;;
        --llm)
            SKIP_EMBEDDING="--skip_embedding"
            ;;
        --large)
            NUM_DOCS=5000
            ;;
        *)
            ;;
    esac
done

echo "Running embedding overhead benchmark..."
echo "  Documents: $NUM_DOCS"
echo "  Output: $OUTPUT"

python analysis/benchmark_embedding_overhead.py \
    --num_docs $NUM_DOCS \
    --batch_size_embedding $BATCH_SIZE_EMBEDDING \
    --batch_size_llm $BATCH_SIZE_LLM \
    --output $OUTPUT \
    $SKIP_LLM \
    $SKIP_EMBEDDING

# Generate visualization if results exist
if [ -f "$OUTPUT" ]; then
    echo ""
    echo "Generating visualization..."
    python analysis/visualize_benchmark.py --input $OUTPUT
fi

