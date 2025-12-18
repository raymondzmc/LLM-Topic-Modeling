module load StdEnv/2023
module load python/3.12.4
module load arrow/21.0.0
module load cuda/12.6

# Pydantic requires rust and cargo
python -m venv ~/virtualenvs/llm-topics
source ~/virtualenvs/llm-topics/bin/activate

pip install -r requirements.txt
pip install flash-attn --no-build-isolation