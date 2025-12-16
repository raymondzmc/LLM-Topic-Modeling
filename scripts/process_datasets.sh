#!/bin/bash
# Process datasets for topic modeling

# tweet_topic.tsv with ERNIE-4.5-0.3B-PT
python data/process_dataset.py \
    --dataset data/raw_data/tweet_topic.tsv \
    --content_key text \
    --label_key label \
    --vocab_size 2000 \
    --model_name baidu/ERNIE-4.5-0.3B-PT \
    --batch_size 64 \
    --embedding_method last \
    --save_name tweet_topic_ERNIE-4.5-0.3B-PT_vocab_2000_last

# tweet_topic.tsv with Llama-3.2-1B-Instruct
python data/process_dataset.py \
    --dataset data/raw_data/tweet_topic.tsv \
    --content_key text \
    --label_key label \
    --vocab_size 2000 \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --batch_size 32 \
    --embedding_method last \
    --save_name tweet_topic_Llama-3.2-1B-Instruct_vocab_2000_last

# tweet_topic.tsv with Llama-3.1-8B-Instruct
python data/process_dataset.py \
    --dataset data/raw_data/tweet_topic.tsv \
    --content_key text \
    --label_key label \
    --vocab_size 2000 \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --batch_size 32 \
    --embedding_method last \
    --save_name tweet_topic_Llama-3.1-8B-Instruct_vocab_2000_last

# stackoverflow.tsv with ERNIE-4.5-0.3B-PT
python data/process_dataset.py \
    --dataset data/raw_data/stackoverflow.tsv \
    --content_key text \
    --label_key label \
    --vocab_size 2000 \
    --model_name baidu/ERNIE-4.5-0.3B-PT \
    --batch_size 32 \
    --embedding_method last \
    --save_name stackoverflow_ERNIE-4.5-0.3B-PT_vocab_2000_last

# stackoverflow.tsv with Llama-3.2-1B-Instruct
python data/process_dataset.py \
    --dataset data/raw_data/stackoverflow.tsv \
    --content_key text \
    --label_key label \
    --vocab_size 2000 \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --batch_size 32 \
    --embedding_method last \
    --save_name stackoverflow_Llama-3.2-1B-Instruct_vocab_2000_last

# stackoverflow.tsv with Llama-3.1-8B-Instruct
python data/process_dataset.py \
    --dataset data/raw_data/stackoverflow.tsv \
    --content_key text \
    --label_key label \
    --vocab_size 2000 \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --batch_size 32 \
    --embedding_method last \
    --save_name stackoverflow_Llama-3.1-8B-Instruct_vocab_2000_last

# SetFit/20_newsgroups with ERNIE-4.5-0.3B-PT
python data/process_dataset.py \
    --dataset SetFit/20_newsgroups \
    --content_key text \
    --label_key label \
    --split all \
    --vocab_size 2000 \
    --model_name baidu/ERNIE-4.5-0.3B-PT \
    --batch_size 32 \
    --embedding_method last \
    --save_name 20_newsgroups_ERNIE-4.5-0.3B-PT_vocab_2000_last

# SetFit/20_newsgroups with Llama-3.2-1B-Instruct
python data/process_dataset.py \
    --dataset SetFit/20_newsgroups \
    --content_key text \
    --label_key label \
    --split all \
    --vocab_size 2000 \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --batch_size 32 \
    --embedding_method last \
    --save_name 20_newsgroups_Llama-3.2-1B-Instruct_vocab_2000_last

# SetFit/20_newsgroups with Llama-3.1-8B-Instruct
python data/process_dataset.py \
    --dataset SetFit/20_newsgroups \
    --content_key text \
    --label_key label \
    --split all \
    --vocab_size 2000 \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --batch_size 32 \
    --embedding_method last \
    --save_name 20_newsgroups_Llama-3.1-8B-Instruct_vocab_2000_last
