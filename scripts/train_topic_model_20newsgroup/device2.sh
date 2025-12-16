#!/usr/bin/env bash
num_topics=75

CUDA_VISIBLE_DEVICES=4 python run_topic_model_baselines.py \
    --model lda \
    --data_path data/20_newsgroups_Llama-3.2-3B-Instruct_vocab_2000_last \
    --results_path results/20_newsgroups/lda_K${num_topics} \
    --num_topics ${num_topics} \
    --num_seeds 5

CUDA_VISIBLE_DEVICES=4 python run_topic_model_baselines.py \
    --model prodlda \
    --data_path data/20_newsgroups_Llama-3.2-3B-Instruct_vocab_2000_last \
    --results_path results/20_newsgroups/prodlda_K${num_topics} \
    --num_topics ${num_topics} \
    --num_hidden_layers 2 \
    --num_seeds 5

CUDA_VISIBLE_DEVICES=4 python run_topic_model_baselines.py \
    --model zeroshot \
    --data_path data/20_newsgroups_Llama-3.2-3B-Instruct_vocab_2000_last \
    --results_path results/20_newsgroups/zeroshot_K${num_topics} \
    --num_topics ${num_topics} \
    --num_hidden_layers 2 \
    --num_seeds 5

CUDA_VISIBLE_DEVICES=4 python run_topic_model_baselines.py \
    --model combined \
    --data_path data/20_newsgroups_Llama-3.2-3B-Instruct_vocab_2000_last \
    --results_path results/20_newsgroups/combined_K${num_topics} \
    --num_topics ${num_topics} \
    --num_hidden_layers 2 \
    --num_seeds 5

CUDA_VISIBLE_DEVICES=4 python run_topic_model_baselines.py \
    --model etm \
    --data_path data/20_newsgroups_Llama-3.2-3B-Instruct_vocab_2000_last \
    --results_path results/20_newsgroups/etm_K${num_topics} \
    --num_topics ${num_topics} \
    --num_hidden_layers 2 \
    --num_seeds 5

CUDA_VISIBLE_DEVICES=4 python run_topic_model_ours.py \
    --data_path data/20_newsgroups_Llama-3.2-3B-Instruct_vocab_2000_last \
    --results_path results/20_newsgroups/Llama-3.2-3B-Instruct/${num_topics}_CE \
    --num_topics ${num_topics} \
    --num_hidden_layers 2 \
    --num_seeds 5 \
    --loss_type CE \
    --temperature 3 \
    --loss_weight 1000

CUDA_VISIBLE_DEVICES=4 python run_topic_model_ours.py \
    --data_path data/20_newsgroups_Llama-3.2-3B-Instruct_vocab_2000_last \
    --results_path results/20_newsgroups/Llama-3.2-3B-Instruct/${num_topics}_KL \
    --num_topics ${num_topics} \
    --num_hidden_layers 2 \
    --num_seeds 5 \
    --loss_type KL \
    --temperature 3 \
    --loss_weight 1000

CUDA_VISIBLE_DEVICES=4 python run_topic_model_ours.py \
    --data_path data/20_newsgroups_Llama-3.2-1B-Instruct_vocab_2000_last \
    --results_path results/20_newsgroups/Llama-3.2-1B-Instruct/${num_topics}_CE \
    --num_topics ${num_topics} \
    --num_hidden_layers 2 \
    --num_seeds 5 \
    --loss_type CE \
    --temperature 3 \
    --loss_weight 1000

CUDA_VISIBLE_DEVICES=4 python run_topic_model_ours.py \
    --data_path data/20_newsgroups_Llama-3.2-1B-Instruct_vocab_2000_last \
    --results_path results/20_newsgroups/Llama-3.2-1B-Instruct/${num_topics}_KL \
    --num_topics ${num_topics} \
    --num_hidden_layers 2 \
    --num_seeds 5 \
    --loss_type KL \
    --temperature 3 \
    --loss_weight 1000
