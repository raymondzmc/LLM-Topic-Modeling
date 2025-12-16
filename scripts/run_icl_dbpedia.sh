python run_icl.py --method zeroshot --output_dir ./results/dbpedia_zeroshot \
                  --dataset_name fancyzhx/dbpedia_14 \
                  --input_key content \
                  --target_key label \
                  --inference_prompt icl_classification_dbpedia.jinja \
                  --evaluation_metric accuracy \
                  --category_names data/dbpedia_label.json

                  
python run_icl.py --method last_hidden_state --output_dir ./results/dbpedia_icl_last_hidden_state \
                  --dataset_name fancyzhx/dbpedia_14 \
                  --input_key content \
                  --target_key label \
                  --inference_prompt icl_classification_dbpedia.jinja \
                  --evaluation_metric accuracy \
                  --category_names data/dbpedia_label.json \
                  --num_demonstrations 3

# python run_icl.py --method topic --output_dir ./results/dbpedia_icl_topic \
#                   --dataset_name fancyzhx/dbpedia_14 \
#                   --topic_model_path data/dbpedia_14_Llama-3.2-1B_2000/checkpoints/generative_K25/run_4 \
#                   --test_dataset_path data/xsum_Llama-3.2-1B_1000_test \
#                   --train_dataset_path data/xsum_Llama-3.2-1B_1000_train \
#                   --input_key content \
#                   --target_key label \
#                   --inference_prompt icl_classification_dbpedia.jinja \
#                   --evaluation_metric accuracy \
#                   --category_names data/dbpedia_label.json \
#                   --num_demonstrations 3