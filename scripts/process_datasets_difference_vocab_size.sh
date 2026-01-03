#!/bin/bash
# Process datasets for topic modeling with different vocabulary sizes
# Model: baidu/ERNIE-4.5-0.3B-PT
# Datasets: tweet_topic, stackoverflow, 20_newsgroups
# Vocab sizes: 500, 1000, 2000, 4000

# tweet_topic.tsv - vocab_size 500
python process_dataset.py \
    --dataset data/raw_data/tweet_topic.tsv \
    --content_key text \
    --label_key label \
    --vocab_size 500 \
    --model_name baidu/ERNIE-4.5-0.3B-PT \
    --batch_size 64 \
    --embedding_method last \
    --save_name tweet_topic_ERNIE-4.5-0.3B-PT_vocab_500_last

# tweet_topic.tsv - vocab_size 1000
python process_dataset.py \
    --dataset data/raw_data/tweet_topic.tsv \
    --content_key text \
    --label_key label \
    --vocab_size 1000 \
    --model_name baidu/ERNIE-4.5-0.3B-PT \
    --batch_size 64 \
    --embedding_method last \
    --save_name tweet_topic_ERNIE-4.5-0.3B-PT_vocab_1000_last

# tweet_topic.tsv - vocab_size 2000
python process_dataset.py \
    --dataset data/raw_data/tweet_topic.tsv \
    --content_key text \
    --label_key label \
    --vocab_size 2000 \
    --model_name baidu/ERNIE-4.5-0.3B-PT \
    --batch_size 64 \
    --embedding_method last \
    --save_name tweet_topic_ERNIE-4.5-0.3B-PT_vocab_2000_last

# tweet_topic.tsv - vocab_size 4000
python process_dataset.py \
    --dataset data/raw_data/tweet_topic.tsv \
    --content_key text \
    --label_key label \
    --vocab_size 4000 \
    --model_name baidu/ERNIE-4.5-0.3B-PT \
    --batch_size 64 \
    --embedding_method last \
    --save_name tweet_topic_ERNIE-4.5-0.3B-PT_vocab_4000_last

# stackoverflow.tsv - vocab_size 500
python process_dataset.py \
    --dataset data/raw_data/stackoverflow.tsv \
    --content_key text \
    --label_key label \
    --vocab_size 500 \
    --model_name baidu/ERNIE-4.5-0.3B-PT \
    --batch_size 64 \
    --embedding_method last \
    --save_name stackoverflow_ERNIE-4.5-0.3B-PT_vocab_500_last

# stackoverflow.tsv - vocab_size 1000
python process_dataset.py \
    --dataset data/raw_data/stackoverflow.tsv \
    --content_key text \
    --label_key label \
    --vocab_size 1000 \
    --model_name baidu/ERNIE-4.5-0.3B-PT \
    --batch_size 64 \
    --embedding_method last \
    --save_name stackoverflow_ERNIE-4.5-0.3B-PT_vocab_1000_last

# stackoverflow.tsv - vocab_size 2000
python process_dataset.py \
    --dataset data/raw_data/stackoverflow.tsv \
    --content_key text \
    --label_key label \
    --vocab_size 2000 \
    --model_name baidu/ERNIE-4.5-0.3B-PT \
    --batch_size 64 \
    --embedding_method last \
    --save_name stackoverflow_ERNIE-4.5-0.3B-PT_vocab_2000_last

# stackoverflow.tsv - vocab_size 4000
python process_dataset.py \
    --dataset data/raw_data/stackoverflow.tsv \
    --content_key text \
    --label_key label \
    --vocab_size 4000 \
    --model_name baidu/ERNIE-4.5-0.3B-PT \
    --batch_size 64 \
    --embedding_method last \
    --save_name stackoverflow_ERNIE-4.5-0.3B-PT_vocab_4000_last

# SetFit/20_newsgroups - vocab_size 500
python process_dataset.py \
    --dataset SetFit/20_newsgroups \
    --content_key text \
    --label_key label \
    --split all \
    --vocab_size 500 \
    --model_name baidu/ERNIE-4.5-0.3B-PT \
    --batch_size 1 \
    --embedding_method last \
    --save_name 20_newsgroups_ERNIE-4.5-0.3B-PT_vocab_500_last

# SetFit/20_newsgroups - vocab_size 1000
python process_dataset.py \
    --dataset SetFit/20_newsgroups \
    --content_key text \
    --label_key label \
    --split all \
    --vocab_size 1000 \
    --model_name baidu/ERNIE-4.5-0.3B-PT \
    --batch_size 1 \
    --embedding_method last \
    --save_name 20_newsgroups_ERNIE-4.5-0.3B-PT_vocab_1000_last

# SetFit/20_newsgroups - vocab_size 2000
python process_dataset.py \
    --dataset SetFit/20_newsgroups \
    --content_key text \
    --label_key label \
    --split all \
    --vocab_size 2000 \
    --model_name baidu/ERNIE-4.5-0.3B-PT \
    --batch_size 1 \
    --embedding_method last \
    --save_name 20_newsgroups_ERNIE-4.5-0.3B-PT_vocab_2000_last

# SetFit/20_newsgroups - vocab_size 4000
python process_dataset.py \
    --dataset SetFit/20_newsgroups \
    --content_key text \
    --label_key label \
    --split all \
    --vocab_size 4000 \
    --model_name baidu/ERNIE-4.5-0.3B-PT \
    --batch_size 1 \
    --embedding_method last \
    --save_name 20_newsgroups_ERNIE-4.5-0.3B-PT_vocab_4000_last

