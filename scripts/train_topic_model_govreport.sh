# #!/usr/bin/env bash
# python process_dataset.py --dataset ccdv/govreport-summarization \
#                           --content_key summary \
#                           --split test \
#                           --vocab_size 2000 \
#                           --model_name meta-llama/Llama-3.2-1B \
#                           --single_token_only \
#                           --batch_size 32 \
#                           --embedding_method last \
#                           --bow_dataset

# python run_topic_modeling.py \
#   --data_path ./data/govreport-summarization_Llama-3.2-1B_vocab_2000_last \
#   --model combined \
#   --K 25 \
#   --test_corpus_path data/govreport-summarization_Llama-3.2-1B_vocab_2000_last/bow_dataset.txt \
#   --num_seeds 3 \
#   --num_epochs 100

# python run_topic_modeling.py \
#   --data_path ./data/govreport-summarization_Llama-3.2-1B_vocab_2000_last \
#   --model combined \
#   --K 50 \
#   --test_corpus_path data/govreport-summarization_Llama-3.2-1B_vocab_2000_last/bow_dataset.txt \
#   --num_seeds 3 \
#   --num_epochs 100

# python run_topic_modeling.py \
#   --data_path ./data/govreport-summarization_Llama-3.2-1B_vocab_2000_last \
#   --model combined \
#   --K 100 \
#   --test_corpus_path data/govreport-summarization_Llama-3.2-1B_vocab_2000_last/bow_dataset.txt \
#   --num_seeds 3 \
#   --num_epochs 100

# python run_topic_modeling.py \
#   --data_path ./data/govreport-summarization_Llama-3.2-1B_vocab_2000_last \
#   --model zeroshot \
#   --K 25 \
#   --test_corpus_path data/govreport-summarization_Llama-3.2-1B_vocab_2000_last/bow_dataset.txt \
#   --num_seeds 3 \
#   --num_epochs 100
  
# python run_topic_modeling.py \
#   --data_path ./data/govreport-summarization_Llama-3.2-1B_vocab_2000_last \
#   --model zeroshot \
#   --K 50 \
#   --test_corpus_path data/govreport-summarization_Llama-3.2-1B_vocab_2000_last/bow_dataset.txt \
#   --num_seeds 3 \
#   --num_epochs 100

# python run_topic_modeling.py \
#   --data_path ./data/govreport-summarization_Llama-3.2-1B_vocab_2000_last \
#   --model zeroshot \
#   --K 100 \
#   --test_corpus_path data/govreport-summarization_Llama-3.2-1B_vocab_2000_last/bow_dataset.txt \
#   --num_seeds 3 \
#   --num_epochs 100

# python run_topic_modeling.py \
#   --data_path ./data/govreport-summarization_Llama-3.2-1B_vocab_2000_last \
#   --model generative \
#   --K 25 \
#   --test_corpus_path data/govreport-summarization_Llama-3.2-1B_vocab_2000_last/bow_dataset.txt \
#   --num_seeds 5 \
#   --hidden_state_layer 16 \
#   --num_epochs 20

python run_topic_modeling.py \
  --data_path ./data/govreport-summarization_Llama-3.2-1B_vocab_2000_last \
  --model generative \
  --K 50 \
  --test_corpus_path data/govreport-summarization_Llama-3.2-1B_vocab_2000_last/bow_dataset.txt \
  --num_seeds 1 \
  --hidden_state_layer 16 \
  --num_epochs 30

python run_topic_modeling.py \
  --data_path ./data/govreport-summarization_Llama-3.2-1B_vocab_2000_last \
  --model generative \
  --K 100 \
  --test_corpus_path data/govreport-summarization_Llama-3.2-1B_vocab_2000_last/bow_dataset.txt \
  --num_seeds 1 \
  --hidden_state_layer 16 \
  --num_epochs 30