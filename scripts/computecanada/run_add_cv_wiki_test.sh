#!/bin/bash
#SBATCH --account=def-carenini
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/run_add_cv_wiki_test.out

# TEST: CPU-only job with DUMMY cv_wiki values
# This tests the wandb upload workflow without actually computing cv_wiki
# Use this to verify the pipeline works before running the full computation

module load StdEnv/2023
module load python/3.12.4
module load arrow/21.0.0
module load java/1.8

source ~/virtualenvs/llm-topics/bin/activate

# Test with a single run using dummy cv_wiki value
echo "Testing with dummy cv_wiki value..."
python scripts/add_missing_cv_wiki.py --run_id 1vsxjr6p --project tweet_topic --dummy

echo ""
echo "Test complete! Check wandb for the new run."
echo "If successful, delete the test run and run the full script:"
echo "  sbatch scripts/computecanada/run_add_cv_wiki.sh"

