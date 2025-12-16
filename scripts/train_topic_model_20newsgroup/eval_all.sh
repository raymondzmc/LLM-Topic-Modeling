#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

# Set this flag to True to only evaluate the model and not train it
EVAL_ONLY=True
RECOMPUTE_METRICS=True

for NUM_TOPICS in 25 50 75 100; do
    python run_topic_model_baselines.py \
        --model lda \
        --data_path data/20_newsgroups_Llama-3.2-3B-Instruct_vocab_2000_last \
        --results_path results/20_newsgroups/lda_K${NUM_TOPICS} \
        --num_topics ${NUM_TOPICS} \
        --num_seeds 5 \
        ${EVAL_ONLY:+--eval_only} \
        ${RECOMPUTE_METRICS:+--recompute_metrics}

    python run_topic_model_baselines.py \
        --model prodlda \
        --data_path data/20_newsgroups_Llama-3.2-3B-Instruct_vocab_2000_last \
        --results_path results/20_newsgroups/prodlda_K${NUM_TOPICS} \
        --num_topics ${NUM_TOPICS} \
        --num_hidden_layers 2 \
        --num_seeds 5 \
        ${EVAL_ONLY:+--eval_only} \
        ${RECOMPUTE_METRICS:+--recompute_metrics}

    python run_topic_model_baselines.py \
        --model zeroshot \
        --data_path data/20_newsgroups_Llama-3.2-3B-Instruct_vocab_2000_last \
        --results_path results/20_newsgroups/zeroshot_K${NUM_TOPICS} \
        --num_topics ${NUM_TOPICS} \
        --num_hidden_layers 2 \
        --num_seeds 5 \
        ${EVAL_ONLY:+--eval_only} \
        ${RECOMPUTE_METRICS:+--recompute_metrics}

    python run_topic_model_baselines.py \
        --model combined \
        --data_path data/20_newsgroups_Llama-3.2-3B-Instruct_vocab_2000_last \
        --results_path results/20_newsgroups/combined_K${NUM_TOPICS} \
        --num_topics ${NUM_TOPICS} \
        --num_hidden_layers 2 \
        --num_seeds 5 \
        ${EVAL_ONLY:+--eval_only} \
        ${RECOMPUTE_METRICS:+--recompute_metrics}

    # python run_topic_model_baselines.py \
    #     --model etm \
    #     --data_path data/20_newsgroups_Llama-3.2-3B-Instruct_vocab_2000_last \
    #     --results_path results/20_newsgroups/etm_K${NUM_TOPICS} \
    #     --num_topics ${NUM_TOPICS} \
    #     --num_hidden_layers 2 \
    #     --num_seeds 5 \
    #     ${EVAL_ONLY:+--eval_only} \
    #     ${RECOMPUTE_METRICS:+--recompute_metrics}

    python run_topic_model_ours.py \
        --data_path data/20_newsgroups_Llama-3.2-3B-Instruct_vocab_2000_last \
        --results_path results/20_newsgroups/Llama-3.2-3B-Instruct/${NUM_TOPICS}_CE \
        --num_topics ${NUM_TOPICS} \
        --num_hidden_layers 2 \
        --num_seeds 5 \
        --loss_type CE \
        --temperature 3 \
        --loss_weight 1000 \
        ${EVAL_ONLY:+--eval_only} \
        ${RECOMPUTE_METRICS:+--recompute_metrics}

    python run_topic_model_ours.py \
        --data_path data/20_newsgroups_Llama-3.2-3B-Instruct_vocab_2000_last \
        --results_path results/20_newsgroups/Llama-3.2-3B-Instruct/${NUM_TOPICS}_KL \
        --num_topics ${NUM_TOPICS} \
        --num_hidden_layers 2 \
        --num_seeds 5 \
        --loss_type KL \
        --temperature 3 \
        --loss_weight 1000 \
        ${EVAL_ONLY:+--eval_only} \
        ${RECOMPUTE_METRICS:+--recompute_metrics}

    python run_topic_model_ours.py \
        --data_path data/20_newsgroups_Llama-3.2-1B-Instruct_vocab_2000_last \
        --results_path results/20_newsgroups/Llama-3.2-1B-Instruct/${NUM_TOPICS}_CE \
        --num_topics ${NUM_TOPICS} \
        --num_hidden_layers 2 \
        --num_seeds 5 \
        --loss_type CE \
        --temperature 3 \
        --loss_weight 1000 \
        ${EVAL_ONLY:+--eval_only} \
        ${RECOMPUTE_METRICS:+--recompute_metrics}

    python run_topic_model_ours.py \
        --data_path data/20_newsgroups_Llama-3.2-1B-Instruct_vocab_2000_last \
        --results_path results/20_newsgroups/Llama-3.2-1B-Instruct/${NUM_TOPICS}_KL \
        --num_topics ${NUM_TOPICS} \
        --num_hidden_layers 2 \
        --num_seeds 5 \
        --loss_type KL \
        --temperature 3 \
        --loss_weight 1000 \
        ${EVAL_ONLY:+--eval_only} \
        ${RECOMPUTE_METRICS:+--recompute_metrics}
done
