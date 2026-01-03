#!/bin/bash
#SBATCH --account=def-carenini
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/run_ablation_temperature_4.out

module load StdEnv/2023
module load python/3.12.4
module load arrow/21.0.0
module load cuda/12.6
module load java/1.8

source ~/virtualenvs/llm-topics/bin/activate
source scripts/run_generative_ablation_temperature_4.sh

