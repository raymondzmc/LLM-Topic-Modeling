export CUDA_VISIBLE_DEVICES=0

# python process_dataset_new.py --dataset SetFit/20_newsgroups \
#                               --content_key text \
#                               --split all \
#                               --vocab_size 2000 \
#                               --model_name meta-llama/Llama-3.2-1B-Instruct \
#                               --hidden_state_layer -1 \
#                               --single_token_only \
#                               --batch_size 32 \
#                               --target_method summary \
#                               --bow_dataset \
#                               --label_key label


for NUM_TOPICS in 25 50 75 100; do
    python run_topic_model_baselines.py \
        --model fastopic \
        --data_path data/20_newsgroups_Llama-3.2-1B-Instruct_vocab_2000_last \
        --results_path results/20_newsgroups/fastopic_K${NUM_TOPICS} \
        --num_topics ${NUM_TOPICS} \
        --num_seeds 5
done
