# #!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python process_dataset_new.py \
  --dataset fancyzhx/dbpedia_14 \
  --content_key content \
  --split test \
  --vocab_size 2000 \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --single_token_only \
  --hidden_state_layer 16 \
  --batch_size 32 \
  --target_method summary \
  --bow_dataset

CUDA_VISIBLE_DEVICES=0 python run_topic_modeling.py \
  --data_path data/dbpedia_14_Llama-3.2-1B-Instruct_vocab_2000_last \
  --model generative \
  --K 50 \
  --hidden_state_layer 16 \
  --test_corpus_path data/dbpedia_14_Llama-3.2-1B-Instruct_vocab_2000_last/bow_dataset.txt \
  --num_seeds 5 \
  --num_epochs 50 
  
CUDA_VISIBLE_DEVICES=0 python run_topic_modeling.py \
  --data_path data/dbpedia_14_Llama-3.2-1B-Instruct_vocab_2000_last \
  --model zeroshot \
  --K 50 \
  --test_corpus_path data/dbpedia_14_Llama-3.2-1B-Instruct_vocab_2000_last/bow_dataset.txt \
  --num_seeds 5 \
  --num_epochs 50
  
CUDA_VISIBLE_DEVICES=1 python run_topic_modeling.py \
  --data_path data/dbpedia_14_Llama-3.2-1B-Instruct_vocab_2000_last \
  --model combined \
  --K 50 \
  --test_corpus_path data/dbpedia_14_Llama-3.2-1B-Instruct_vocab_2000_last/bow_dataset.txt \
  --num_seeds 5 \
  --num_epochs 50

# python run_topic_modeling.py \
#   --data_path data/dbpedia_14_Llama-3.2-1B-Instruct_vocab_2000_last \
#   --model combined \
#   --K 50 \
#   --test_corpus_path data/dbpedia_14_Llama-3.2-1B-Instruct_vocab_2000_last/bow_dataset.txt \
#   --num_seeds 5 \
#   --num_epochs 50 

python run_topic_modeling.py \
  --data_path data/dbpedia_14_Llama-3.2-1B-Instruct_vocab_2000_last \
  --model generative \
  --K 50 \
  --test_corpus_path data/dbpedia_14_Llama-3.2-1B-Instruct_vocab_2000_last/bow_dataset.txt \
  --num_seeds 5 \
  --num_epochs 50 

  
  

# # Hidden layers based on your commands
# HIDDEN_LAYERS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)

# # Loop over hidden layers
# for HL in "${HIDDEN_LAYERS[@]}"; do
#   python run_topic_modeling.py \
#     --data_path data/dbpedia_14_Llama-3.2-1B-Instruct_vocab_2000_last \
#     --model generative \
#     --K 25 \
#     --test_corpus_path data/dbpedia_14_Llama-3.2-1B-Instruct_vocab_2000_last/bow_dataset.txt \
#     --num_seeds 5 \
#     --num_epochs 20 

#   python run_topic_modeling.py \
#     --data_path data/dbpedia_14_Llama-3.2-1B-Instruct_vocab_2000_last \
#     --model generative \
#     --K 25 \
#     --test_corpus_path data/dbpedia_14_Llama-3.2-1B-Instruct_vocab_2000_last/bow_dataset.txt \
#     --num_seeds 5 \
#     --num_epochs 50 
# done