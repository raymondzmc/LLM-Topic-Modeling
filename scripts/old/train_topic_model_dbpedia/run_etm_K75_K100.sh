export CUDA_VISIBLE_DEVICES=4

for NUM_TOPICS in 100; do
    python run_topic_model_baselines.py \
        --model etm \
        --data_path data/dbpedia_14_Llama-3.2-3B-Instruct_vocab_2000_last \
        --results_path results/dbpedia_14/etm_K${NUM_TOPICS} \
        --num_topics ${NUM_TOPICS} \
        --num_seeds 5
done
