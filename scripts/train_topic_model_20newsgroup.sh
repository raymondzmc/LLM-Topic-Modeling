CUDA_VISIBLE_DEVICES=3 python process_dataset_new.py --dataset SetFit/20_newsgroups \
                          --content_key text \
                          --split all \
                          --vocab_size 2000 \
                          --model_name meta-llama/Llama-3.2-1B-Instruct \
                          --hidden_state_layer -1 \
                          --single_token_only \
                          --batch_size 32 \
                          --target_method summary \
                          --bow_dataset \
                          --label_key label

python run_topic_modeling.py \
  --data_path ./data/20_newsgroups_Llama-3.2-1B_vocab_2000_last \
  --model combined \
  --K 50 \
  --hidden_state_layer 16 \
  --test_corpus_path data/20_newsgroups_Llama-3.2-1B_vocab_2000_last/bow_dataset.txt \
  --num_epochs 30 \
  --num_seeds 1

python run_topic_modeling.py \
    --data_path ./data/20_newsgroups_Llama-3.2-1B_vocab_2000_last \
    --model generative \
    --K 50 \
    --hidden_state_layer 16 \
    --test_corpus_path data/20_newsgroups_Llama-3.2-1B_vocab_2000_last/bow_dataset.txt \
    --num_epochs 30 \
    --num_seeds 1

# python run_topic_modeling.py \
#   --data_path ./data/20_newsgroups_Llama-3.2-1B_vocab_2000_last \
#   --model zeroshot \
#   --K 50 \
#   --test_corpus_path data/20_newsgroups_Llama-3.2-1B_vocab_2000_last/bow_dataset.txt \
#   --num_seeds 5 \
#   --num_epochs 50


# python run_topic_modeling.py \
#     --data_path ./data/20_newsgroups_Llama-3.2-1B_vocab_2000_last \
#     --model generative \
#     --K 100 \
#     --hidden_state_layer 16 \
#     --num_epochs 50 \
#     --test_corpus_path data/20_newsgroups_Llama-3.2-1B_vocab_2000_last/bow_dataset.txt \
#     --num_seeds 5

# python run_topic_modeling.py \
#   --data_path ./data/20_newsgroups_Llama-3.2-1B_vocab_2000_last \
#   --model zeroshot \
#   --K 100 \
#   --test_corpus_path data/20_newsgroups_Llama-3.2-1B_vocab_2000_last/bow_dataset.txt \
#   --num_seeds 5 \
#   --num_epochs 50

# python run_topic_modeling.py \
#   --data_path ./data/20_newsgroups_Llama-3.2-1B_vocab_2000_last \
#   --model combined \
#   --K 100 \
#   --test_corpus_path data/20_newsgroups_Llama-3.2-1B_vocab_2000_last/bow_dataset.txt \
#   --num_seeds 5 \
#   --num_epochs 50