#!/bin/bash
#SBATCH --account=def-carenini
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/computational_runtime.out

module load StdEnv/2023
module load python/3.12.4
module load arrow/21.0.0
module load cuda/12.6

source ~/virtualenvs/llm-topics/bin/activate
python analysis/benchmark_embedding_overhead.py
