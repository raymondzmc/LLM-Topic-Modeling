#!/usr/bin/env bash
python process_dataset.py --dataset EdinburghNLP/xsum \
                          --content_key document \
                          --split test \
                          --vocab_size 1000 \
                          --model_name meta-llama/Llama-3.2-1B \
                          --single_token_only \
                          --hidden_state_layer -1 \
                          --bow_dataset 


# python run_topic_modeling.py \
#   --data_path ./data/govreport-summarization_Llama-3.2-1B_1000 \
#   --model zeroshot \
#   --K 25 \
#   --test_corpus_path data/govreport-summarization_Llama-3.2-1B_1000/bow_dataset.txt \
#   --num_seeds 3 \
#   --num_epochs 100
  
# python run_topic_modeling.py \
#   --data_path ./data/govreport-summarization_Llama-3.2-1B_1000 \
#   --model zeroshot \
#   --K 50 \
#   --test_corpus_path data/govreport-summarization_Llama-3.2-1B_1000/bow_dataset.txt \
#   --num_seeds 3 \
#   --num_epochs 100
  
# python run_topic_modeling.py \
#   --data_path ./data/govreport-summarization_Llama-3.2-1B_1000 \
#   --model combined \
#   --K 25 \
#   --test_corpus_path data/govreport-summarization_Llama-3.2-1B_1000/bow_dataset.txt \
#   --num_seeds 3 \
#   --num_epochs 100

# python run_topic_modeling.py \
#   --data_path ./data/govreport-summarization_Llama-3.2-1B_1000 \
#   --model combined \
#   --K 50 \
#   --test_corpus_path data/govreport-summarization_Llama-3.2-1B_1000/bow_dataset.txt \
#   --num_seeds 3 \
#   --num_epochs 100 

python run_topic_modeling.py \
  --data_path ./data/xsum_Llama-3.2-1B_1000 \
  --model generative \
  --K 25 \
  --test_corpus_path data/xsum_Llama-3.2-1B_1000/bow_dataset.txt \
  --num_seeds 1 \
  --num_epochs 100

python run_topic_modeling.py \
  --data_path ./data/xsum_Llama-3.2-1B_1000\
  --model generative \
  --K 50 \
  --test_corpus_path data/xsum_Llama-3.2-1B_1000/bow_dataset.txt \
  --num_seeds 1 \
  --num_epochs 100 
