module load StdEnv/2023
module load python/3.12.4
module load arrow/21.0.0
module load cuda/12.6
source ~/venv/llm-topics/bin/activate

# Pydantic requires rust and cargo
export HF_HUB_CACHE=/home/liraymo6/projects/def-carenini/liraymo6/LLM-topics/.cache