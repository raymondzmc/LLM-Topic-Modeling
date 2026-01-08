#!/bin/bash
# Run missing temperature ablation experiments for stackoverflow
# Model: ERNIE-4.5-0.3B-PT
# Missing temperatures: 0.5, 1.0, 2.0, 6.0, 7.0, 9.0 (all K)
# Plus: 4.0 (K=75, K=100), 5.0 (K=100), 8.0 (K=100)

model="ERNIE-4.5-0.3B-PT"
dataset="stackoverflow"
DATA_PATH="raymondzmc/${dataset}_${model}_vocab_2000_last"

# Missing temperatures for all K values
temperatures_all_k=(
    0.5
    1.0
    2.0
    6.0
    7.0
    9.0
)

for K in 25 50 75 100
do
    for temp in "${temperatures_all_k[@]}"
    do
        echo "Running Generative TM on $DATA_PATH with K=$K, temperature=$temp"
        python run_topic_model.py \
            --model generative \
            --data_path "$DATA_PATH" \
            --num_topics $K \
            --temperature $temp
    done
done

# Additional missing/crashed runs
echo "Running additional missing/crashed runs..."

# temp=4.0, K=75 (crashed)
echo "Running Generative TM on $DATA_PATH with K=75, temperature=4.0"
python run_topic_model.py \
    --model generative \
    --data_path "$DATA_PATH" \
    --num_topics 75 \
    --temperature 4.0

# temp=4.0, K=100 (missing)
echo "Running Generative TM on $DATA_PATH with K=100, temperature=4.0"
python run_topic_model.py \
    --model generative \
    --data_path "$DATA_PATH" \
    --num_topics 100 \
    --temperature 4.0

# temp=5.0, K=100 (missing)
echo "Running Generative TM on $DATA_PATH with K=100, temperature=5.0"
python run_topic_model.py \
    --model generative \
    --data_path "$DATA_PATH" \
    --num_topics 100 \
    --temperature 5.0

# temp=8.0, K=100 (crashed)
echo "Running Generative TM on $DATA_PATH with K=100, temperature=8.0"
python run_topic_model.py \
    --model generative \
    --data_path "$DATA_PATH" \
    --num_topics 100 \
    --temperature 8.0

