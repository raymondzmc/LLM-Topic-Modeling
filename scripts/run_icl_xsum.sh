# python run_icl.py --method mean_hidden_states --output_dir ./results/xsum_icl_mean_hidden_states

# Process training set for topic model
# python process_dataset.py --dataset EdinburghNLP/xsum \
#                           --data_path data \
#                           --dir_name xsum_Llama-3.2-1B_1000_train \
#                           --content_key document \
#                           --vocab_path data/xsum_Llama-3.2-1B_1000_test/vocab.json \
#                           --model_name meta-llama/Llama-3.2-1B \
#                           --split train \
#                           --use_flash_attention_2 \
#                           --single_token_only \
#                           --hidden_state_layer -1 \
#                           --bow_dataset \
#                           --id_key id \
#                           --max_tokens 512
                        
# python run_icl.py --method topic --output_dir ./results/xsum_icl_topic --dataset_name EdinburghNLP/xsum \
#                   --topic_model_path data/xsum_Llama-3.2-1B_1000_test/checkpoints/generative_K25/run_0 \
#                   --test_dataset_path data/xsum_Llama-3.2-1B_1000_test \
#                   --train_dataset_path data/xsum_Llama-3.2-1B_1000_train
                  
python run_icl.py --method last_hidden_state --output_dir ./results/xsum_icl_last_hidden_state \
                  --dataset_name EdinburghNLP/xsum 