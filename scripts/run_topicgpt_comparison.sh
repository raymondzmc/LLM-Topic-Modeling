# stackoverflow.tsv with ERNIE-4.5-0.3B-PT
python process_dataset.py \
    --dataset data/raw_data/stackoverflow_train.tsv \
    --content_key text \
    --label_key label \
    --vocab_size 2000 \
    --model_name baidu/ERNIE-4.5-0.3B-PT \
    --batch_size 64 \
    --embedding_method last \
    --save_name stackoverflow_train_ERNIE-4.5-0.3B-PT_vocab_2000_last

# stackoverflow.tsv with Llama-3.2-1B-Instruct
python process_dataset.py \
    --dataset data/raw_data/stackoverflow_train.tsv \
    --content_key text \
    --label_key label \
    --vocab_size 2000 \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --batch_size 64 \
    --embedding_method last \
    --save_name stackoverflow_train_Llama-3.2-1B-Instruct_vocab_2000_last

# stackoverflow.tsv with Llama-3.1-8B-Instruct
python process_dataset.py \
    --dataset data/raw_data/stackoverflow_train.tsv \
    --content_key text \
    --label_key label \
    --vocab_size 2000 \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --batch_size 1 \
    --embedding_method last \
    --save_name stackoverflow_train_Llama-3.1-8B-Instruct_vocab_2000_last

model=(
    "ERNIE-4.5-0.3B-PT"
    "Llama-3.2-1B-Instruct"
    "Llama-3.1-8B-Instruct"
)

for model in "${models[@]}"
do
    python run_topic_model.py \
        --model generative \
        --data_path "raymondzmc/stackoverflow_train_${model}_vocab_2000_last" \
        --num_topics 22
done
