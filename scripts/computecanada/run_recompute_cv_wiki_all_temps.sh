#!/bin/bash
#SBATCH --account=def-carenini
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/run_recompute_cv_wiki_all_temps.out

# CPU-only job for recomputing cv_wiki using Palmetto (Java-based)
# Processes ALL temperature ablation runs (80 runs total)
# Uses --force to re-compute even if cv_wiki already exists

module load StdEnv/2023
module load python/3.12.4
module load arrow/21.0.0
module load java/1.8

source ~/virtualenvs/llm-topics/bin/activate

# Run the recompute script
bash scripts/recompute_cv_wiki_all_temps.sh

