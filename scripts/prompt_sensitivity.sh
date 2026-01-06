variants=(
    "variant_1"
    "variant_2"
    "variant_3"
    "variant_4"
    "variant_5"
)

for variant in "${variants[@]}"
do
    python process_dataset.py \
        --dataset SetFit/20_newsgroups \
        --content_key text \
        --label_key label \
        --split all \
        --vocab_size 2000 \
        --model_name baidu/ERNIE-4.5-0.3B-PT \
        --batch_size 1 \
        --embedding_method last \
        --save_name 20_newsgroups_ERNIE-4.5-0.3B-PT_vocab_2000_last_${variant} \
        --instruction_template instructions/${variant}.jinja

    DATA_PATH="raymondzmc/20_newsgroups_ERNIE-4.5-0.3B-PT_vocab_2000_last_${variant}"

    for K in 50
    do
        python run_topic_model.py \
            --model generative \
            --data_path "$DATA_PATH" \
            --num_topics $K \
            --wandb_project "prompt_sensitivity"
        
        python run_topic_model.py \
            --model zeroshot \
            --data_path "$DATA_PATH" \
            --num_topics $K \
            --wandb_project "prompt_sensitivity"

        python run_topic_model.py \
            --model prodlda \
            --data_path "$DATA_PATH" \
            --num_topics $K \
            --wandb_project "prompt_sensitivity"
        
        python run_topic_model.py \
            --model fastopic \
            --data_path "$DATA_PATH" \
            --num_topics $K \
            --wandb_project "prompt_sensitivity"
    done
done

