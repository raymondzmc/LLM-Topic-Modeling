#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0

# python process_dataset_new.py \
#     --dataset data/stackoverflow.csv \
#     --content_key text \
#     --label_key label \
#     --split all \
#     --vocab_size 2000 \
#     --model_name meta-llama/Llama-3.2-1B-Instruct \
#     --hidden_state_layer -1 \
#     --single_token_only \
#     --batch_size 32 \
#     --target_method summary \
#     --bow_dataset

# python process_dataset_new.py \
#     --dataset data/stackoverflow.csv \
#     --content_key text \
#     --label_key label \
#     --split all \
#     --vocab_size 2000 \
#     --model_name meta-llama/Llama-3.2-3B-Instruct \
#     --hidden_state_layer -1 \
#     --single_token_only \
#     --batch_size 32 \
#     --target_method summary \
#     --bow_dataset

CUDA_VISIBLE_DEVICES=0 python process_dataset_new.py \
    --dataset data/stackoverflow.csv \
    --content_key text \
    --label_key label \
    --split all \
    --vocab_size 2000 \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --dir_name stackoverflow_Llama-3.2-1B-Instruct_vocab_2000_last_variant_1 \
    --hidden_state_layer -1 \
    --single_token_only \
    --batch_size 32 \
    --target_method summary \
    --instruction_template instructions/variant_1.jinja \
    --bow_dataset

CUDA_VISIBLE_DEVICES=1 python process_dataset_new.py \
    --dataset data/stackoverflow.csv \
    --content_key text \
    --label_key label \
    --split all \
    --vocab_size 2000 \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --dir_name stackoverflow_Llama-3.2-1B-Instruct_vocab_2000_last_variant_2 \
    --hidden_state_layer -1 \
    --single_token_only \
    --batch_size 32 \
    --target_method summary \
    --instruction_template instructions/variant_2.jinja \
    --bow_dataset

CUDA_VISIBLE_DEVICES=2 python process_dataset_new.py \
    --dataset data/stackoverflow.csv \
    --content_key text \
    --label_key label \
    --split all \
    --vocab_size 2000 \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --dir_name stackoverflow_Llama-3.2-1B-Instruct_vocab_2000_last_variant_3 \
    --hidden_state_layer -1 \
    --single_token_only \
    --batch_size 32 \
    --target_method summary \
    --instruction_template instructions/variant_3.jinja \
    --bow_dataset

CUDA_VISIBLE_DEVICES=3 python process_dataset_new.py \
    --dataset data/stackoverflow.csv \
    --content_key text \
    --label_key label \
    --split all \
    --vocab_size 2000 \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --dir_name stackoverflow_Llama-3.2-1B-Instruct_vocab_2000_last_variant_4 \
    --hidden_state_layer -1 \
    --single_token_only \
    --batch_size 32 \
    --target_method summary \
    --instruction_template instructions/variant_4.jinja \
    --bow_dataset

python process_dataset_new.py \
    --dataset data/stackoverflow.csv \
    --content_key text \
    --label_key label \
    --split all \
    --vocab_size 2000 \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --dir_name stackoverflow_Llama-3.2-1B-Instruct_vocab_2000_last_variant_5 \
    --hidden_state_layer -1 \
    --single_token_only \
    --batch_size 32 \
    --target_method summary \
    --prompt_template instructions/variant_5.jinja \
    --bow_dataset

