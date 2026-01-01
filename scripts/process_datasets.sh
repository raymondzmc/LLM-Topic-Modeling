python process_dataset.py \
    --dataset data/raw_data/stackoverflow.tsv \
    --content_key text \
    --label_key label \
    --vocab_size 2000 \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --batch_size 1 \
    --embedding_method last \
    --save_name stackoverflow_Llama-3.1-8B-Instruct_vocab_2000_last

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

python process_dataset.py \
    --dataset SetFit/20_newsgroups \
    --content_key text \
    --label_key label \
    --split all \
    --vocab_size 2000 \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --batch_size 1 \
    --embedding_method last \
    --save_name 20_newsgroups_Llama-3.1-8B-Instruct_vocab_2000_last
