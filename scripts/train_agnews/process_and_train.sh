#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=5
python process_dataset_new.py \
    --dataset fancyzhx/ag_news \
    --content_key text \
    --split all \
    --vocab_size 2000 \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --hidden_state_layer -1 \
    --single_token_only \
    --batch_size 32 \
    --target_method summary \
    --bow_dataset

python process_dataset_new.py \
    --dataset fancyzhx/ag_news \
    --content_key text \
    --split all \
    --vocab_size 2000 \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --hidden_state_layer -1 \
    --single_token_only \
    --batch_size 32 \
    --target_method summary \
    --bow_dataset

for num_topics in 25 50 75 100; do
    python run_topic_model_baselines.py \
        --model lda \
        --data_path data/ag_news_Llama-3.2-3B-Instruct_vocab_2000_last \
        --results_path results/ag_news/lda_K${num_topics} \
        --num_topics ${num_topics} \
        --num_seeds 5

    python run_topic_model_baselines.py \
        --model prodlda \
        --data_path data/ag_news_Llama-3.2-3B-Instruct_vocab_2000_last \
        --results_path results/ag_news/prodlda_K${num_topics} \
        --num_topics ${num_topics} \
        --num_hidden_layers 2 \
        --num_seeds 5

    python run_topic_model_baselines.py \
        --model zeroshot \
        --data_path data/ag_news_Llama-3.2-3B-Instruct_vocab_2000_last \
        --results_path results/ag_news/zeroshot_K${num_topics} \
        --num_topics ${num_topics} \
        --num_hidden_layers 2 \
        --num_seeds 5

    python run_topic_model_baselines.py \
        --model combined \
        --data_path data/ag_news_Llama-3.2-3B-Instruct_vocab_2000_last \
        --results_path results/ag_news/combined_K${num_topics} \
        --num_topics ${num_topics} \
        --num_hidden_layers 2 \
        --num_seeds 5

    python run_topic_model_baselines.py \
        --model etm \
        --data_path data/ag_news_Llama-3.2-3B-Instruct_vocab_2000_last \
        --results_path results/ag_news/etm_K${num_topics} \
        --num_topics ${num_topics} \
        --num_hidden_layers 2 \
        --num_seeds 5

    python run_topic_model_ours.py \
        --data_path data/ag_news_Llama-3.2-3B-Instruct_vocab_2000_last \
        --results_path results/ag_news/Llama-3.2-3B-Instruct/${num_topics}_CE \
        --num_topics ${num_topics} \
        --num_hidden_layers 2 \
        --num_seeds 5 \
        --loss_type CE \
        --temperature 3 \
        --loss_weight 1000

    python run_topic_model_ours.py \
        --data_path data/ag_news_Llama-3.2-3B-Instruct_vocab_2000_last \
        --results_path results/ag_news/Llama-3.2-3B-Instruct/${num_topics}_KL \
        --num_topics ${num_topics} \
        --num_hidden_layers 2 \
        --num_seeds 5 \
        --loss_type KL \
        --temperature 3 \
        --loss_weight 1000

    python run_topic_model_ours.py \
        --data_path data/ag_news_Llama-3.2-1B-Instruct_vocab_2000_last \
        --results_path results/ag_news/Llama-3.2-1B-Instruct/${num_topics}_CE \
        --num_topics ${num_topics} \
        --num_hidden_layers 2 \
        --num_seeds 5 \
        --loss_type CE \
        --temperature 3 \
        --loss_weight 1000

    python run_topic_model_ours.py \
        --data_path data/ag_news_Llama-3.2-1B-Instruct_vocab_2000_last \
        --results_path results/ag_news/Llama-3.2-1B-Instruct/${num_topics}_KL \
        --num_topics ${num_topics} \
        --num_hidden_layers 2 \
        --num_seeds 5 \
        --loss_type KL \
        --temperature 3 \
        --loss_weight 1000
done