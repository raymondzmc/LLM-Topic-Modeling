#!/bin/bash
#SBATCH --account=def-carenini
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/run_add_cv_wiki.out

# CPU-only job for computing cv_wiki using Palmetto (Java-based)
# No GPU required

module load StdEnv/2023
module load python/3.12.4
module load arrow/21.0.0
module load java/1.8

source ~/virtualenvs/llm-topics/bin/activate

# Run the cv_wiki addition script for all missing runs
python scripts/add_missing_cv_wiki.py --all

