#!/usr/bin/env bash

for num_topics in 25 50 100; do
  for temperature in 1 2 3 5 10; do
    CUDA_VISIBLE_DEVICES=3 python run_topic_model_ours.py \
      --data_path data/dbpedia_14_Llama-3.2-3B-Instruct_vocab_2000_last \
      --results_path results/dbpedia/Llama-3.2-3B-Instruct/K${num_topics}_KL_temperature${temperature}_loss_weight1000 \
      --num_topics ${num_topics} \
      --num_hidden_layers 2 \
      --num_seeds 5 \
      --loss_type KL \
      --temperature ${temperature} \
      --loss_weight 1000
  done
done