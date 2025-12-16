#!/bin/bash
#SBATCH --account=def-carenini
#SBATCH --time=2-00:00:00
#SBATCH --gpus-per-node=v100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/process_20newsgroup_llama32.out
module load StdEnv/2023 arrow/15 python/3.10

# Pydantic requires rust and cargo
module load rust/1.70.0 
source ~/virtualenvs/llm-topics/bin/activate
export HF_HUB_CACHE=/home/liraymo6/projects/def-carenini/liraymo6/LLM-topics/.cache
python process_dataset.py --dataset google-research-datasets/newsgroup \
                          --content_key text \
                          --split train \
                          --vocab_path data/20newsgroup_vocab.json \
                          --model_name meta-llama/Llama-3.2-1B \
                          --data_path data \
                          --batch_size 32 \
                          --examples_per_vocab 100
