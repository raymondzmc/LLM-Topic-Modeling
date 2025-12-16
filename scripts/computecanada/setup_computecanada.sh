module load StdEnv/2023 arrow/15 python/3.10

# Pydantic requires rust and cargo
module load rust/1.70.0 
source ~/virtualenvs/llm-topics/bin/activate
export HF_HUB_CACHE=/home/liraymo6/projects/def-carenini/liraymo6/LLM-topics/.cache