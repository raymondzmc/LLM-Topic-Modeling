"""Unified script for training and evaluating topic models."""

import os
import json
import time
import argparse
import random
import numpy as np
import torch
import wandb

from gensim.downloader import load as gensim_load
from octis.dataset.dataset import Dataset as OCTISDataset
from octis.models.LDA import LDA
from octis.models.ProdLDA import ProdLDA
from octis.models.CTM import CTM
from octis.models.ETM import ETM
from bertopic import BERTopic
from topmost.data import RawDataset

from data.loaders import load_training_data, prepare_octis_files, PROCESSED_DATA_DIR
from models.ctm import CTM as GenerativeTM
from utils.dataset import get_ctm_dataset_generative
from utils.metrics import compute_aggregate_results, evaluate_topic_model
from utils.embeddings import get_openai_embedding
from utils.fastopic_trainer import FASTopicTrainer
from settings import settings


# Models that require processed LLM data (embeddings + logits)
LLM_MODELS = {'generative'}

# Baseline models that use BOW data
BASELINE_MODELS = {'lda', 'prodlda', 'zeroshot', 'combined', 'etm', 'bertopic', 'fastopic'}

ALL_MODELS = LLM_MODELS | BASELINE_MODELS


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def prepare_octis_dataset(data_path: str, bow_corpus: list[list[str]], vocab: list[str]) -> OCTISDataset:
    """Prepare OCTIS dataset format from BOW corpus."""
    # Ensure OCTIS files exist
    prepare_octis_files(data_path, bow_corpus, vocab)
    
    dataset = OCTISDataset()
    dataset.load_custom_dataset_from_folder(data_path)
    return dataset


def train_model(
    model_name: str,
    args: argparse.Namespace,
    seed: int,
    seed_dir: str,
    local_data_path: str,
    vocab: list[str],
    bow_corpus: list[list[str]],
    processed_dataset: dict = None,
    octis_dataset: OCTISDataset = None,
) -> dict:
    """Train a topic model.
    
    Args:
        model_name: Name of the model to train
        args: Command line arguments
        seed: Random seed
        seed_dir: Directory to save seed-specific outputs
        local_data_path: Path to local data directory
        vocab: Vocabulary list
        bow_corpus: Bag of words corpus
        processed_dataset: Processed dataset for generative models (embeddings + logits)
        octis_dataset: OCTIS dataset for baseline models
        
    Returns:
        Model output dictionary containing topics and topic-document matrix
    """
    # Generative (LLM-based) model
    if model_name == 'generative':
        if processed_dataset is None:
            raise ValueError("Generative model requires processed_dataset with embeddings and logits")
        
        ctm_dataset = get_ctm_dataset_generative(processed_dataset, vocab)
        
        model = GenerativeTM(
            input_size=len(vocab),
            bert_input_size=ctm_dataset.X_contextual.shape[1],
            num_topics=args.num_topics,
            activation=args.activation,
            hidden_sizes=tuple([args.hidden_size] * args.num_hidden_layers),
            solver=args.solver,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            loss_weight=args.loss_weight,
            sparsity_ratio=args.sparsity_ratio,
            loss_type=args.loss_type,
            temperature=args.temperature,
            top_words=args.top_words,
        )
        model.fit(ctm_dataset)
        return model.get_info()
    
    # LDA
    elif model_name == 'lda':
        model = LDA(num_topics=args.num_topics, random_state=seed)
        return model.train_model(dataset=octis_dataset, top_words=args.top_words)
    
    # ProdLDA
    elif model_name == 'prodlda':
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
        return model.train_model(dataset=octis_dataset, top_words=args.top_words)
    
    # CTM (ZeroShot or Combined)
    elif model_name in ['zeroshot', 'combined']:
        model = CTM(
            num_topics=args.num_topics,
            num_layers=args.num_hidden_layers,
            num_neurons=args.hidden_size,
            batch_size=args.batch_size,
            lr=args.lr,
            activation=args.activation,
            solver=args.solver,
            num_epochs=args.num_epochs,
            inference_type=model_name,
            bert_path=os.path.join(local_data_path, 'bert'),
            bert_model='all-mpnet-base-v2',
            use_partitions=False,
        )
        model.set_seed(seed)
        return model.train_model(dataset=octis_dataset, top_words=args.top_words)
    
    # ETM
    elif model_name == 'etm':
        word2vec_path = 'word2vec-google-news-300.kv'
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
        return model.train_model(
            dataset=octis_dataset,
            top_words=args.top_words,
            op_path=os.path.join(seed_dir, 'checkpoint.pt'),
        )
    
    # BERTopic
    elif model_name == 'bertopic':
        model = BERTopic(
            language='english',
            top_n_words=args.top_words,
            nr_topics=args.num_topics + 1,
            calculate_probabilities=True,
            verbose=True,
            low_memory=False,
        )
        text_corpus = [' '.join(word_list) for word_list in bow_corpus]
        output = model.fit_transform(text_corpus)
        all_topics = model.get_topics()
        topics = [
            [word_prob[0] for word_prob in topic]
            for topic_id, topic in all_topics.items() if topic_id != -1
        ]
        return {
            'topics': topics,
            'topic-document-matrix': output[1].transpose(),
        }
    
    # FASTopic
    elif model_name == 'fastopic':
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
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            'topics': [topic_string.split(' ') for topic_string in top_words],
            'topic-document-matrix': doc_topic_dist.transpose(),
        }
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run(args: argparse.Namespace):
    """Main training and evaluation loop."""
    # Load training data
    is_generative = args.model in LLM_MODELS
    training_data = load_training_data(
        args.data_path,
        local_name=args.local_name,
        for_generative=is_generative,
    )
    
    # Extract dataset name from local path
    dataset_name = os.path.basename(training_data.local_path).split('_')[0]
    
    # Determine results path
    if args.results_path is None:
        args.results_path = f'results/{dataset_name}/{args.model}_K{args.num_topics}'
    
    os.makedirs(args.results_path, exist_ok=True)
    
    # Prepare OCTIS dataset for baseline models
    octis_dataset = None
    if args.model in BASELINE_MODELS:
        octis_dataset = prepare_octis_dataset(
            training_data.local_path,
            training_data.bow_corpus,
            training_data.vocab,
        )
    
    # Load or compute vocab embeddings for evaluation
    vocab_embedding_path = os.path.join(training_data.local_path, 'vocab_embeddings.json')
    if os.path.exists(vocab_embedding_path):
        with open(vocab_embedding_path, encoding='utf-8') as f:
            vocab_embeddings = json.load(f)
    elif training_data.vocab is not None:
        vocab_embeddings = get_openai_embedding(training_data.vocab)
        os.makedirs(os.path.dirname(vocab_embedding_path), exist_ok=True)
        with open(vocab_embedding_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_embeddings, f)
    else:
        vocab_embeddings = None
    
    # Initialize wandb with dataset name as project
    wandb_project = args.wandb_project if args.wandb_project else dataset_name
    wandb_config = {
        'model': args.model,
        'dataset': dataset_name,
        'num_topics': args.num_topics,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'hidden_size': args.hidden_size,
        'num_hidden_layers': args.num_hidden_layers,
        'activation': args.activation,
        'solver': args.solver,
        'top_words': args.top_words,
        'num_seeds': args.num_seeds,
    }
    if args.model in LLM_MODELS:
        wandb_config.update({
            'loss_weight': args.loss_weight,
            'sparsity_ratio': args.sparsity_ratio,
            'loss_type': args.loss_type,
            'temperature': args.temperature,
        })
    
    wandb.init(
        project=wandb_project,
        entity=settings.wandb_entity,
        name=f"{args.model}_K{args.num_topics}",
        config=wandb_config,
        mode='online' if not args.wandb_offline else 'offline',
    )
    
    all_results = []
    
    for seed in range(args.num_seeds):
        set_seed(seed)
        seed_dir = os.path.join(args.results_path, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        
        model_output_path = os.path.join(seed_dir, 'model_output.pt')
        
        # Check if model output already exists
        if os.path.exists(model_output_path):
            model_output = torch.load(model_output_path, weights_only=False)
            if model_output.get('topic-document-matrix') is not None:
                print(f"[Seed {seed}] Loading existing model output from {model_output_path}")
                training_time = model_output.get('training_time', 0)
            else:
                model_output = None
                training_time = 0
        else:
            model_output = None
            training_time = 0
        
        # Train model if needed
        if model_output is None:
            if args.eval_only:
                raise ValueError(
                    f"Model output does not exist in \"{seed_dir}\" when eval_only is True. "
                    "Please re-run the script without --eval_only"
                )
            
            print(f"\n[Seed {seed}] Training {args.model} model...")
            start_time = time.time()
            
            model_output = train_model(
                model_name=args.model,
                args=args,
                seed=seed,
                seed_dir=seed_dir,
                local_data_path=training_data.local_path,
                vocab=training_data.vocab,
                bow_corpus=training_data.bow_corpus,
                processed_dataset=training_data.processed_dataset,
                octis_dataset=octis_dataset,
            )
            
            training_time = time.time() - start_time
            model_output['training_time'] = training_time
            
            print(f"[Seed {seed}] Training completed in {training_time:.2f} seconds")
            
            # Save model output
            torch.save(model_output, model_output_path)
        
        # Save topics
        topics_path = os.path.join(seed_dir, 'topics.json')
        if not os.path.exists(topics_path):
            with open(topics_path, 'w', encoding='utf-8') as f:
                json.dump(model_output['topics'], f)
        
        # Evaluate model
        eval_results_path = os.path.join(seed_dir, 'evaluation_results.json')
        if not os.path.exists(eval_results_path) or args.recompute_metrics:
            print(f"\n[Seed {seed}] Evaluating model...")
            evaluation_results = evaluate_topic_model(
                model_output,
                top_words=args.top_words,
                test_corpus=training_data.bow_corpus,
                embeddings=vocab_embeddings,
                labels=training_data.labels,
            )
            evaluation_results['training_time'] = training_time
            
            with open(eval_results_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f)
        else:
            with open(eval_results_path, encoding='utf-8') as f:
                evaluation_results = json.load(f)
        
        # Log to wandb
        wandb_log = {f"seed_{seed}/{k}": v for k, v in evaluation_results.items()}
        wandb_log[f"seed_{seed}/training_time"] = training_time
        wandb.log(wandb_log)
        
        # Upload artifacts for this seed
        artifact = wandb.Artifact(
            name=f"{args.model}_K{args.num_topics}_seed{seed}",
            type="model",
            description=f"Model output for {args.model} with {args.num_topics} topics (seed {seed})"
        )
        artifact.add_file(model_output_path)
        artifact.add_file(topics_path)
        artifact.add_file(eval_results_path)
        wandb.log_artifact(artifact)
        
        all_results.append(evaluation_results)
    
    # Compute and save aggregated results
    averaged_results = compute_aggregate_results(args.results_path)
    averaged_results_path = os.path.join(args.results_path, 'averaged_results.json')
    with open(averaged_results_path, 'w', encoding='utf-8') as f:
        json.dump(averaged_results, f)
    
    # Log averaged results to wandb
    wandb.log({f"avg/{k}": v for k, v in averaged_results.items()})
    
    # Upload final aggregated results artifact
    final_artifact = wandb.Artifact(
        name=f"{args.model}_K{args.num_topics}_results",
        type="results",
        description=f"Aggregated results for {args.model} with {args.num_topics} topics"
    )
    final_artifact.add_dir(args.results_path)
    wandb.log_artifact(final_artifact)
    
    wandb.finish()
    
    print(f"\nResults saved to: {args.results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate topic models")
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data directory or HF repo ID')
    parser.add_argument('--local_name', type=str, default=None,
                        help='Local name for processed dataset cache')
    parser.add_argument('--results_path', type=str, default=None,
                        help='Path to save results')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='generative',
                        choices=list(ALL_MODELS),
                        help='Model to train')
    parser.add_argument('--num_topics', type=int, default=25,
                        help='Number of topics')
    parser.add_argument('--top_words', type=int, default=20,
                        help='Number of top words per topic')
    
    # Training arguments
    parser.add_argument('--num_seeds', type=int, default=5,
                        help='Number of random seeds to run')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-3,
                        help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=200,
                        help='Hidden layer size')
    parser.add_argument('--num_hidden_layers', type=int, default=2,
                        help='Number of hidden layers')
    parser.add_argument('--activation', type=str, default='softplus',
                        help='Activation function')
    parser.add_argument('--solver', type=str, default='adam',
                        help='Optimizer')
    
    # Generative model specific arguments
    parser.add_argument('--loss_weight', type=float, default=1.0,
                        help='Weight for reconstruction loss (generative only)')
    parser.add_argument('--sparsity_ratio', type=float, default=1.0,
                        help='Sparsity ratio for teacher logits (generative only)')
    parser.add_argument('--loss_type', type=str, default='KL', choices=['KL', 'CE'],
                        help='Loss type (generative only)')
    parser.add_argument('--temperature', type=float, default=3.0,
                        help='Temperature for softmax (generative only)')
    
    # Evaluation arguments
    parser.add_argument('--eval_only', action='store_true',
                        help='Only evaluate existing models')
    parser.add_argument('--recompute_metrics', action='store_true',
                        help='Recompute evaluation metrics')
    
    # Wandb arguments
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='Wandb project name (default: dataset name)')
    parser.add_argument('--wandb_offline', action='store_true',
                        help='Run wandb in offline mode')
    
    args = parser.parse_args()
    run(args)
