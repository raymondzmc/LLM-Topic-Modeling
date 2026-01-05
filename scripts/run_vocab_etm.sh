#!/bin/bash
# Run ETM for vocab size experiments

for vocab_size in 500 1000 4000
do
    DATA_PATH="raymondzmc/20_newsgroups_ERNIE-4.5-0.3B-PT_vocab_${vocab_size}_last"
    for K in 25 50 75 100
    do
        echo "Running ETM on $DATA_PATH with K=$K"
        python run_topic_model.py \
            --model etm \
            --data_path "$DATA_PATH" \
            --num_topics $K \
            --wandb_project "20_newsgroups_vocab_${vocab_size}"
    done
done

