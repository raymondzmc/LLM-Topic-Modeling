# #!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python process_dataset_new.py \
  --dataset fancyzhx/dbpedia_14 \
  --content_key content \
  --split all \
  --vocab_size 2000 \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --hidden_state_layer -1 \
  --single_token_only \
  --batch_size 32 \
  --target_method summary \
  --bow_dataset \
  --label_key label

CUDA_VISIBLE_DEVICES=0 python run_topic_model_baselines.py \
  --data_path data/dbpedia_14_Llama-3.2-3B-Instruct_vocab_2000_last \
  --model zeroshot \
  --num_topics 25 \
  --num_hidden_layers 2 \
  --num_seeds 5

# CUDA_VISIBLE_DEVICES=1 python run_topic_model_baselines.py \
#   --data_path data/dbpedia_14_Llama-3.2-1B-Instruct_vocab_4000_last \
#   --model zeroshot \
#   --num_topics 50 \
#   --num_hidden_layers 2 \
#   --num_seeds 5

# CUDA_VISIBLE_DEVICES=2 python run_topic_model_baselines.py \
#   --data_path data/dbpedia_14_Llama-3.2-1B-Instruct_vocab_4000_last \
#   --model zeroshot \
#   --num_topics 100 \
#   --num_hidden_layers 2 \
#   --num_seeds 5


# CUDA_VISIBLE_DEVICES=0 python run_topic_model_baselines.py \
#   --data_path data/dbpedia_14_Llama-3.2-1B-Instruct_vocab_4000_last \
#   --model combined \
#   --num_topics 25 \
#   --num_hidden_layers 2 \
#   --num_seeds 5

# CUDA_VISIBLE_DEVICES=1 python run_topic_model_baselines.py \
#   --data_path data/dbpedia_14_Llama-3.2-1B-Instruct_vocab_4000_last \
#   --model combined \
#   --num_topics 50 \
#   --num_hidden_layers 2 \
#   --num_seeds 5

# CUDA_VISIBLE_DEVICES=2 python run_topic_model_baselines.py \
#   --data_path data/dbpedia_14_Llama-3.2-1B-Instruct_vocab_4000_last \
#   --model combined \
#   --num_topics 100 \
#   --num_hidden_layers 2 \
#   --num_seeds 5

# # Ours

CUDA_VISIBLE_DEVICES=0 python run_topic_model_ours.py \
  --data_path data/dbpedia_14_Llama-3.2-3B-Instruct_vocab_2000_last \
  --num_topics 50 \
  --num_hidden_layers 2 \
  --num_seeds 5

# CUDA_VISIBLE_DEVICES=4 python run_topic_model_ours.py \
#   --data_path data/dbpedia_14_Llama-3.2-1B-Instruct_vocab_4000_last \
#   --num_topics 50 \
#   --num_hidden_layers 2 \
#   --num_seeds 5

# CUDA_VISIBLE_DEVICES=5 python run_topic_model_ours.py \
#   --data_path data/dbpedia_14_Llama-3.2-1B-Instruct_vocab_4000_last \
#   --num_topics 100 \
#   --num_hidden_layers 2 \
#   --num_seeds 5

# # Loss weight experiment
# CUDA_VISIBLE_DEVICES=0 python run_topic_model_ours.py \
#   --data_path data/dbpedia_14_Llama-3.2-1B-Instruct_vocab_4000_last \
#   --results_path results/dbpedia/generative_K50_sparsity1e-3 \
#   --num_topics 50 \
#   --num_hidden_layers 2 \
#   --num_seeds 5 \
#   --num_epochs 50 \
#   --sparsity_ratio 0.01 \
#   --loss_weight 0.1

# CUDA_VISIBLE_DEVICES=1 python run_topic_model_ours.py \
#   --data_path data/dbpedia_14_Llama-3.2-1B-Instruct_vocab_4000_last \
#   --results_path results/dbpedia/generative_K50_sparsity1e-3 \
#   --num_topics 25 \
#   --num_hidden_layers 2 \
#   --num_seeds 5 \
#   --num_epochs 50 \
#   --sparsity_ratio 0.01 \
#   --loss_weight 0.1

# CUDA_VISIBLE_DEVICES=2 python run_topic_model_ours.py \
#   --data_path data/dbpedia_14_Llama-3.2-1B-Instruct_vocab_4000_last \
#   --results_path results/dbpedia/generative_K50_sparsity1e-3 \
#   --num_topics 100 \
#   --num_hidden_layers 2 \
#   --num_seeds 5 \
#   --num_epochs 50 \
#   --sparsity_ratio 0.01 \
#   --loss_weight 0.1