export CUDA_VISIBLE_DEVICES=1

for NUM_TOPICS in 25 50 75 100; do
    python run_topic_model_baselines.py \
        --model bertopic \
        --data_path data/stackoverflow_Llama-3.2-3B-Instruct_vocab_2000_last \
        --results_path results/stackoverflow/bertopic_K${NUM_TOPICS} \
        --num_topics ${NUM_TOPICS} \
        --num_seeds 5
done
