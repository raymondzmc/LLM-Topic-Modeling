import os
import json
import argparse
from utils.io import load_processed_dataset
from utils.embeddings import get_openai_embedding
from utils.metrics import compute_aggregate_results
from models.ctm import CTM as GenerativeTM
from utils.metrics import evaluate_topic_model
import torch


def run(args):
    if args.results_path is None:
        dataset_name = os.path.basename(args.data_path).split('_')[0]
        args.results_path = f'results/{dataset_name}/generative_K{args.num_topics}'

    os.makedirs(args.results_path, exist_ok=True)

    with open(os.path.join(args.data_path, 'bow_dataset.txt'), 'r', encoding='utf-8') as f:
        bow_corpus = [line.strip().split() for line in f if line.strip()]

    processed_dataset = None
    if not args.eval_only:
        processed_data_path = os.path.join(args.data_path, 'processed_dataset')
        processed_dataset = load_processed_dataset(processed_data_path, layer_idx=args.input_hidden_layer)
        vocab_path = os.path.join(args.data_path, 'vocab.json')
        processed_dataset['vocab'] = json.load(open(vocab_path, encoding='utf-8'))

    vocab_embedding_path = os.path.join(args.data_path, 'vocab_embeddings.json')
    if os.path.exists(vocab_embedding_path):
        vocab_embeddings = json.load(open(vocab_embedding_path, encoding='utf-8'))
    elif processed_dataset is not None:
        vocab_embeddings = get_openai_embedding(processed_dataset['vocab'])
        with open(vocab_embedding_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_embeddings, f)
    else:
        vocab_embeddings = None
    
    if os.path.exists(os.path.join(args.data_path, 'numeric_labels.txt')):
        with open(os.path.join(args.data_path, 'numeric_labels.txt'), 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f if line.strip()]
    else:
        labels = None

    for seed in range(5):
        seed_dir = os.path.join(args.results_path, f"seed_{seed}")
        if not os.path.exists(seed_dir):
            os.makedirs(seed_dir)

        model_output_path = os.path.join(seed_dir, 'model_output.pt')
        if os.path.exists(model_output_path):
            model_output = torch.load(model_output_path, weights_only=False)
        else:
            assert not args.eval_only, \
                (f"Model output does not exist in \"{seed_dir}\" when eval_only set is True,"
                 " please re-run the script without --eval_only")

            model = GenerativeTM(
                num_topics=args.num_topics,
                activation=args.activation,
                num_layers=args.num_hidden_layers,
                num_neurons=args.hidden_size,
                solver=args.solver,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=seed,
                loss_weight=args.loss_weight,
                sparsity_ratio=args.sparsity_ratio,
                loss_type=args.loss_type,
                temperature=args.temperature,
                use_partitions=False,
            )
            model.train_model(dataset=processed_dataset, top_words=args.top_words)
            model_output = model.model.get_info()
            torch.save(model_output, os.path.join(seed_dir, 'model_output.pt'))
        topics = model_output['topics']
        if not os.path.exists(os.path.join(seed_dir, 'topics.json')):
            with open(os.path.join(seed_dir, 'topics.json'), 'w', encoding="utf-8") as f:
                json.dump(topics, f)

        if not os.path.exists(os.path.join(seed_dir, 'evaluation_results.json')):
            evaluation_results = evaluate_topic_model(model_output, top_words=args.top_words, test_corpus=bow_corpus, embeddings=vocab_embeddings, labels=labels)
            with open(os.path.join(seed_dir, 'evaluation_results.json'), 'w', encoding="utf-8") as f:
                json.dump(evaluation_results, f)
        elif args.recompute_metrics:
            evaluation_results = json.load(open(os.path.join(seed_dir, 'evaluation_results.json'), encoding='utf-8'))
            new_results = evaluate_topic_model(model_output, top_words=args.top_words, test_corpus=bow_corpus, embeddings=vocab_embeddings, labels=labels)
            evaluation_results.update(new_results)
            with open(os.path.join(seed_dir, 'evaluation_results.json'), 'w', encoding="utf-8") as f:
                json.dump(evaluation_results, f)
        else:
            # Nothing to do here since metrics have already been computed
            pass

    averaged_results = compute_aggregate_results(args.results_path)
    with open(os.path.join(args.results_path, 'averaged_results.json'), 'w', encoding='utf-8') as f:
        json.dump(averaged_results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--results_path', type=str, default=None)
    parser.add_argument('--num_topics', type=int, default=25, help='Number of topics')
    parser.add_argument('--input_hidden_layer', type=int, default=-1)
    parser.add_argument('--solver', type=str, default='adam')
    parser.add_argument('--activation', type=str, default='softplus')
    parser.add_argument('--num_hidden_layers', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=200)
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--top_words', type=int, default=20)
    parser.add_argument('--loss_weight', type=float, default=1.0)
    parser.add_argument('--sparsity_ratio', type=float, default=1.0)
    parser.add_argument('--loss_type', type=str, default='KL', choices=['KL', 'CE'])
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--recompute_metrics', action='store_true')
    args = parser.parse_args()
    run(args)
