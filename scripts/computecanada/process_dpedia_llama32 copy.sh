#!/bin/bash
#SBATCH --account=def-carenini
#SBATCH --time=2-00:00:00
#SBATCH --gpus-per-node=v100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/process_dpedia_llama32.out
module load StdEnv/2023 arrow/15 python/3.10

# Pydantic requires rust and cargo
module load rust/1.70.0 
source ~/virtualenvs/llm-topics/bin/activate
export HF_HUB_CACHE=/home/liraymo6/projects/def-carenini/liraymo6/LLM-topics/.cache
python process_dataset.py --dataset fancyzhx/dbpedia_14 \
                          --content_key content \
                          --split test \
                          --vocab_size 2000 \
                          --examples_per_vocab 200 \
                          --single_token_only  \
                          --model_name meta-llama/Llama-3.2-1B \
                          --data_path data \
                          --batch_size 8
