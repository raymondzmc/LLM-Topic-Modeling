import os
import csv
import json
import argparse
import random
import numpy as np
from gensim.downloader import load as gensim_load
from octis.dataset.dataset import Dataset
from octis.models.LDA import LDA
from octis.models.ProdLDA import ProdLDA
from octis.models.CTM import CTM
from octis.models.ETM import ETM
from bertopic import BERTopic
from utils.metrics import compute_aggregate_results
from utils.metrics import evaluate_topic_model
from utils.embeddings import get_openai_embedding
from utils.fastopic_trainer import FASTopicTrainer
from topmost.data import RawDataset

import torch
import pdb


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def run(args):
    if args.results_path is None:
        dataset_name = os.path.basename(args.data_path).split('_')[0]
        args.results_path = f'results/{dataset_name}/{args.model}_K{args.num_topics}'

    os.makedirs(args.results_path, exist_ok=True)

    with open(os.path.join(args.data_path, 'bow_dataset.txt'), 'r', encoding='utf-8') as f:
        bow_corpus = [line.strip().split() for line in f]
        ignore_indices = [i for i, doc in enumerate(bow_corpus) if len(doc) == 0 or doc == ['null']]
        bow_corpus = [doc for i, doc in enumerate(bow_corpus) if i not in ignore_indices]

    # Prepare the corpus and vocabulary for OCTIS dataset
    corpus_file = os.path.join(args.data_path, 'corpus.tsv')
    vocab_file = os.path.join(args.data_path, 'vocab.txt')
    labels_file = os.path.join(args.data_path, 'numeric_labels.txt')
    if not os.path.exists(corpus_file):
        with open(corpus_file, 'w', encoding='utf-8', newline='') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t')
            for doc in bow_corpus:
                # Each row: document, partition, label (empty)
                writer.writerow([' '.join(doc), 'train', ''])

    if not os.path.exists(vocab_file):
        with open(os.path.join(args.data_path, 'vocab.json'), encoding="utf-8") as f:
            vocab = json.load(f)

        with open(os.path.join(args.data_path, "vocabulary.txt"), "w", encoding="utf-8") as vocab_file:
            for word in vocab:
                vocab_file.write(f"{word}\n")

    if os.path.exists(labels_file):
        with open(labels_file, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f if line.strip()]
        labels = [label for i, label in enumerate(labels) if i not in ignore_indices]
    else:
        labels = None

    dataset = Dataset()
    dataset.load_custom_dataset_from_folder(args.data_path)

    # Compute OpenAI embeddings for evaluation
    vocab_embedding_path = os.path.join(args.data_path, 'vocab_embeddings.json')
    if os.path.exists(vocab_embedding_path):
        vocab_embeddings = json.load(open(vocab_embedding_path, encoding='utf-8'))
    else:
        vocab_embeddings = get_openai_embedding(vocab)
        with open(vocab_embedding_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_embeddings, f)
    
    for seed in range(args.num_seeds):
        set_seed(seed)
        seed_dir = os.path.join(args.results_path, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        model_output_path = os.path.join(seed_dir, 'model_output.pt')
        if os.path.exists(model_output_path) and torch.load(model_output_path, weights_only=False).get('topic-document-matrix') is not None:
            model_output = torch.load(model_output_path, weights_only=False)
        else:
            if args.model == 'lda':
                model = LDA(
                    num_topics=args.num_topics,
                    random_state=seed,
                )
                model_output = model.train_model(dataset=dataset, top_words=args.top_words)
            elif args.model == 'prodlda':
                model = ProdLDA(
                    num_topics=args.num_topics,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    activation=args.activation,
                    solver=args.solver,
                    num_layers=args.num_hidden_layers,
                    num_neurons=args.hidden_size,
                    num_epochs=args.num_epochs,
                    use_partitions=False,
                )
                model_output = model.train_model(dataset=dataset, top_words=args.top_words)
            elif args.model in ['zeroshot', 'combined']:
                model = CTM(
                    num_topics=args.num_topics,
                    num_layers=args.num_hidden_layers,
                    num_neurons=args.hidden_size,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    activation=args.activation,
                    solver=args.solver,
                    num_epochs=args.num_epochs,
                    inference_type=args.model,
                    bert_path=os.path.join(args.data_path, 'bert'),
                    bert_model='all-mpnet-base-v2',
                    use_partitions=False,
                )
                model.set_seed(seed)
                model_output = model.train_model(dataset=dataset, top_words=args.top_words)
            elif args.model == 'etm':
                word2vec_path = os.path.join('word2vec-google-news-300.kv')
                if not os.path.exists(word2vec_path):
                    word2vec = gensim_load('word2vec-google-news-300')
                    word2vec.save_word2vec_format(word2vec_path, binary=True)

                model = ETM(
                    num_topics=args.num_topics,
                    use_partitions=False,
                    train_embeddings=False,
                    embeddings_path=word2vec_path,
                    embeddings_type='word2vec',
                    binary_embeddings=True,
                )
                model_output = model.train_model(
                    dataset=dataset,
                    top_words=args.top_words,
                    op_path=os.path.join(seed_dir, 'checkpoint.pt'),
                )
            elif args.model == 'bertopic':
                model = BERTopic(
                    language='english',
                    top_n_words=args.top_words,
                    nr_topics=args.num_topics+1,
                    calculate_probabilities=True,
                    verbose=True,
                    low_memory=False,
                )
                text_corpus = [' '.join(word_list) for word_list in bow_corpus]
                output = model.fit_transform(text_corpus)
                all_topics = model.get_topics()
                topics = [[word_prob[0] for word_prob in topic] for topic_id, topic in all_topics.items() if topic_id != -1]  
                model_output = {
                    'topics': topics,
                    'topic-document-matrix': output[1].transpose(),
                }
            elif args.model == 'fastopic':
                text_corpus = [' '.join(word_list) for word_list in bow_corpus]
                device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                dataset = RawDataset(text_corpus, device=device)
                trainer = FASTopicTrainer(
                    dataset=dataset,
                    num_topics=args.num_topics,
                    num_top_words=args.top_words,
                    low_memory=True,
                    low_memory_batch_size=262144,
                )
                
                top_words, doc_topic_dist = trainer.train()
                
                # Clear GPU memory after training
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                model_output = {
                    'topics': [topic_string.split(' ') for topic_string in top_words],
                    'topic-document-matrix': doc_topic_dist.transpose(),
                }
            else:
                raise ValueError(f"Model {args.model} not supported")
            torch.save(model_output, os.path.join(seed_dir, 'model_output.pt'))

        if not os.path.exists(os.path.join(seed_dir, 'topics.json')):
            with open(os.path.join(seed_dir, 'topics.json'), 'w', encoding="utf-8") as f:
                json.dump(model_output['topics'], f)

        # TODO: Can be deleted after
        if len(model_output['topics'][0]) != args.top_words:
            print(f"Number of top words in model output: {len(model_output['topics'][0])}")
            print(f"Number of top topic words expected: {args.top_words}")
            topics = json.load(open(os.path.join(seed_dir, 'topics.json'), encoding='utf-8'))
            if len(topics[0]) == args.top_words:
                print("Saved topics have the correct number of top words")
                model_output['topics'] = topics
                torch.save(model_output, os.path.join(seed_dir, 'model_output.pt'))

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
    parser.add_argument('--model', type=str, default='zeroshot', choices=['zeroshot', 'combined', 'prodlda', 'lda', 'etm', 'bertopic', 'fastopic', 'ecrtm'])
    parser.add_argument('--num_topics', type=int, default=25, help='Number of topics')
    parser.add_argument('--num_hidden_layers', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=200)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--top_words', type=int, default=20)
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--activation', type=str, default='softplus')
    parser.add_argument('--solver', type=str, default='adam')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--recompute_metrics', action='store_true')
    args = parser.parse_args()
    run(args)
