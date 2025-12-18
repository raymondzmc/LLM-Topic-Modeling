"""Unified script for training and evaluating topic models."""

import os
import json
import time
import argparse
import random
import tempfile
import numpy as np
import torch
import wandb

from gensim.downloader import load as gensim_load
from data.dataset import OCTISDataset
from models.octis.LDA import LDA
from models.octis.ProdLDA import ProdLDA
from models.octis.CTM import CTM
from models.octis.ETM import ETM
from models.fastopic import FASTopicTrainer
from bertopic import BERTopic
from topmost.data import RawDataset
from data.loaders import load_training_data, prepare_octis_files
from data.dataset.ctm_dataset import get_ctm_dataset_from_processed_data
from models.ctm import CTM as GenerativeTM
from evaluation_metrics.metrics import compute_aggregate_results, evaluate_topic_model
from utils.embeddings import get_openai_embedding


from settings import settings


LLM_MODELS = {'generative'}
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
    prepare_octis_files(data_path, bow_corpus, vocab)
    dataset = OCTISDataset()
    dataset.load_custom_dataset_from_folder(data_path)
    return dataset


def train_model(
    model_name: str,
    args: argparse.Namespace,
    seed: int,
    checkpoint_dir: str,
    local_data_path: str,
    vocab: list[str],
    bow_corpus: list[list[str]],
    processed_data: dict = None,
    octis_dataset: OCTISDataset = None,
) -> dict:
    """Train a topic model and return output dictionary."""
    if model_name == 'generative':
        if processed_data is None:
            raise ValueError("Generative model requires processed_data with embeddings and logits")
        
        ctm_dataset = get_ctm_dataset_from_processed_data(processed_data, vocab)
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
    
    elif model_name == 'lda':
        model = LDA(num_topics=args.num_topics, random_state=seed)
        return model.train_model(dataset=octis_dataset, top_words=args.top_words)
    
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
            op_path=os.path.join(checkpoint_dir, 'checkpoint.pt'),
        )
    
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


def run_reevaluate(args: argparse.Namespace):
    """Re-evaluate a model from a previous W&B run."""
    if args.wandb_project is None:
        raise ValueError("--wandb_project is required when using --load_run_id_or_name")
    
    run_id_or_name = args.load_run_id_or_name
    wandb_project = args.wandb_project
    
    print(f"\n{'='*60}")
    print(f"Re-evaluating from W&B run: {run_id_or_name}")
    print(f"Project: {settings.wandb_entity}/{wandb_project}")
    print(f"{'='*60}\n")
    
    api = wandb.Api()
    
    # Find run by ID first, then by name
    source_run = None
    try:
        source_run = api.run(f"{settings.wandb_entity}/{wandb_project}/{run_id_or_name}")
        print(f"Found run by ID: {source_run.name} ({source_run.id})")
    except wandb.errors.CommError:
        print(f"Run ID '{run_id_or_name}' not found, searching by name...")
        runs = api.runs(
            f"{settings.wandb_entity}/{wandb_project}",
            filters={"display_name": run_id_or_name},
            order="-created_at",
        )
        runs_list = list(runs)
        
        if len(runs_list) == 0:
            raise ValueError(f"No run found with ID or name: {run_id_or_name}")
        
        if len(runs_list) > 1:
            print(f"⚠️  WARNING: Found {len(runs_list)} runs with name '{run_id_or_name}', using most recent")
            for i, r in enumerate(runs_list[:5]):
                print(f"   {i+1}. ID: {r.id}, Created: {r.created_at}")
        
        source_run = runs_list[0]
        print(f"Using run: {source_run.name} ({source_run.id})")
    
    # Find model artifact
    print("\nSearching for model artifact...")
    artifacts = list(source_run.logged_artifacts())
    model_artifacts = [a for a in artifacts if a.type == "model"]
    
    if len(model_artifacts) == 0:
        raise ValueError(f"No model artifacts found for run {source_run.id}")
    
    artifact = model_artifacts[-1]
    print(f"Found artifact: {artifact.name} (v{artifact.version})")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_dir = artifact.download(root=temp_dir)
        print(f"Downloaded to: {artifact_dir}")
        
        # Load labels
        labels_path = os.path.join(artifact_dir, 'labels.json')
        labels = None
        if os.path.exists(labels_path):
            with open(labels_path, encoding='utf-8') as f:
                labels = json.load(f)
            print(f"Loaded {len(labels)} labels")
        
        # Get metadata
        metadata = artifact.metadata or {}
        num_seeds = metadata.get('num_seeds', 1)
        top_words = args.top_words or metadata.get('top_words', 10)
        model_name = metadata.get('model', 'unknown')
        dataset_name = metadata.get('dataset', 'unknown')
        num_topics = metadata.get('num_topics', 0)
        
        print(f"\nMetadata: model={model_name}, dataset={dataset_name}, K={num_topics}, seeds={num_seeds}")
        
        # Initialize new run
        new_run = wandb.init(
            project=wandb_project,
            entity=settings.wandb_entity,
            name=f"{model_name}_K{num_topics}_reevaluate",
            config={
                "source_run_id": source_run.id,
                "source_run_name": source_run.name,
                "model": model_name,
                "dataset": dataset_name,
                "num_topics": num_topics,
                "num_seeds": num_seeds,
                "top_words": top_words,
                "reevaluation": True,
            },
            mode='online' if not args.wandb_offline else 'offline',
        )
        
        results_dir = os.path.join(temp_dir, 'reevaluated')
        os.makedirs(results_dir, exist_ok=True)
        
        for seed in range(num_seeds):
            seed_dir = os.path.join(artifact_dir, f"seed_{seed}")
            new_seed_dir = os.path.join(results_dir, f"seed_{seed}")
            os.makedirs(new_seed_dir, exist_ok=True)
            
            model_output_path = os.path.join(seed_dir, 'model_output.pt')
            if not os.path.exists(model_output_path):
                print(f"[Seed {seed}] model_output.pt not found, skipping")
                continue
            
            print(f"[Seed {seed}] Re-evaluating...")
            model_output = torch.load(model_output_path, weights_only=False)
            training_time = model_output.get('training_time', 0)
            
            # Copy model output
            torch.save(model_output, os.path.join(new_seed_dir, 'model_output.pt'))
            
            # Copy topics
            if 'topics' in model_output:
                with open(os.path.join(new_seed_dir, 'topics.json'), 'w', encoding='utf-8') as f:
                    json.dump(model_output['topics'], f)
            
            # Re-evaluate
            evaluation_results = evaluate_topic_model(
                model_output,
                top_words=top_words,
                test_corpus=None,
                embeddings=None,
                labels=labels,
            )
            evaluation_results['training_time'] = training_time
            
            with open(os.path.join(new_seed_dir, 'evaluation_results.json'), 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f)
            
            print(f"[Seed {seed}] {evaluation_results}")
            new_run.log({f"seed_{seed}/{k}": v for k, v in evaluation_results.items()})
        
        # Aggregated results
        averaged_results = compute_aggregate_results(results_dir)
        with open(os.path.join(results_dir, 'averaged_results.json'), 'w', encoding='utf-8') as f:
            json.dump(averaged_results, f)
        
        # Copy labels and vocab embeddings
        if labels is not None:
            with open(os.path.join(results_dir, 'labels.json'), 'w', encoding='utf-8') as f:
                json.dump(labels, f)
        
        new_run.log({f"avg/{k}": v for k, v in averaged_results.items()})
        print(f"\nAveraged: {averaged_results}")
        
        # Upload artifact
        has_labels = labels is not None
        new_artifact = wandb.Artifact(
            name=f"{model_name}-K{num_topics}-{dataset_name}",
            type="model",
            description=f"Re-evaluated from {source_run.name} ({source_run.id})",
            metadata={
                "model": model_name,
                "dataset": dataset_name,
                "num_topics": num_topics,
                "num_seeds": num_seeds,
                "top_words": top_words,
                "has_labels": has_labels,
                "source_run_id": source_run.id,
                "reevaluation": True,
            }
        )
        new_artifact.add_dir(results_dir)
        new_run.log_artifact(new_artifact)
        new_run.finish()
        
        print(f"\n{'='*60}")
        print("Re-evaluation complete!")
        print(f"View: https://wandb.ai/{settings.wandb_entity}/{wandb_project}")
        print(f"{'='*60}")


def run(args: argparse.Namespace):
    """Main training and evaluation loop."""
    is_generative = args.model in LLM_MODELS
    training_data = load_training_data(args.data_path, for_generative=is_generative)
    dataset_name = os.path.basename(training_data.local_path).split('_')[0]
    
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
    
    # Build config
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
    
    wb_run = wandb.init(
        project=wandb_project,
        entity=settings.wandb_entity,
        name=f"{args.model}_K{args.num_topics}",
        config=wandb_config,
        mode='online' if not args.wandb_offline else 'offline',
    )
    
    # Use temp directory for all outputs (upload to wandb only)
    with tempfile.TemporaryDirectory() as results_dir:
        all_results = []
        
        for seed in range(args.num_seeds):
            set_seed(seed)
            seed_dir = os.path.join(results_dir, f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)
            
            print(f"\n[Seed {seed}] Training {args.model}...")
            start_time = time.time()
            
            model_output = train_model(
                model_name=args.model,
                args=args,
                seed=seed,
                checkpoint_dir=seed_dir,
                local_data_path=training_data.local_path,
                vocab=training_data.vocab,
                bow_corpus=training_data.bow_corpus,
                processed_data=training_data.processed_dataset,
                octis_dataset=octis_dataset,
            )
            
            training_time = time.time() - start_time
            model_output['training_time'] = training_time
            print(f"[Seed {seed}] Trained in {training_time:.2f}s")
            
            # Save model output
            torch.save(model_output, os.path.join(seed_dir, 'model_output.pt'))
            
            # Save topics
            with open(os.path.join(seed_dir, 'topics.json'), 'w', encoding='utf-8') as f:
                json.dump(model_output['topics'], f)
            
            # Evaluate
            print(f"[Seed {seed}] Evaluating...")
            evaluation_results = evaluate_topic_model(
                model_output,
                top_words=args.top_words,
                test_corpus=training_data.bow_corpus,
                embeddings=vocab_embeddings,
                labels=training_data.labels,
            )
            evaluation_results['training_time'] = training_time
            
            with open(os.path.join(seed_dir, 'evaluation_results.json'), 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f)
            
            wb_run.log({f"seed_{seed}/{k}": v for k, v in evaluation_results.items()})
            all_results.append(evaluation_results)
        
        # Aggregated results
        averaged_results = compute_aggregate_results(results_dir)
        with open(os.path.join(results_dir, 'averaged_results.json'), 'w', encoding='utf-8') as f:
            json.dump(averaged_results, f)
        
        # Save labels for re-evaluation
        has_labels = training_data.labels is not None
        if has_labels:
            labels_list = training_data.labels
            if hasattr(labels_list, 'tolist'):
                labels_list = labels_list.tolist()
            with open(os.path.join(results_dir, 'labels.json'), 'w', encoding='utf-8') as f:
                json.dump(labels_list, f)
        
        # Save vocab for reproducibility
        if training_data.vocab is not None:
            with open(os.path.join(results_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
                json.dump(training_data.vocab, f)
        
        # Save config for reproducibility
        with open(os.path.join(results_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(wandb_config, f)
        
        wb_run.log({f"avg/{k}": v for k, v in averaged_results.items()})
        
        # Upload artifact
        print("\nUploading artifact to wandb...")
        artifact = wandb.Artifact(
            name=f"{args.model}-K{args.num_topics}-{dataset_name}",
            type="model",
            description=f"Topic model: {args.model} on {dataset_name} (K={args.num_topics}, seeds={args.num_seeds})",
            metadata={
                "model": args.model,
                "dataset": dataset_name,
                "num_topics": args.num_topics,
                "num_seeds": args.num_seeds,
                "top_words": args.top_words,
                "has_labels": has_labels,
            }
        )
        artifact.add_dir(results_dir)
        
        total_size_mb = sum(
            os.path.getsize(os.path.join(dp, fn))
            for dp, _, fns in os.walk(results_dir) for fn in fns
        ) / (1024 * 1024)
        print(f"  Artifact: {total_size_mb:.2f} MB")
        
        wb_run.log_artifact(artifact)
        wb_run.finish()
        
        print(f"\nView run: https://wandb.ai/{settings.wandb_entity}/{wandb_project}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate topic models")
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to data directory or HF repo ID')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='generative',
                        choices=list(ALL_MODELS), help='Model to train')
    parser.add_argument('--num_topics', type=int, default=25, help='Number of topics')
    parser.add_argument('--top_words', type=int, default=10, help='Top words per topic')
    
    # Training arguments
    parser.add_argument('--num_seeds', type=int, default=5, help='Number of random seeds')
    parser.add_argument('--num_epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=200, help='Hidden layer size')
    parser.add_argument('--num_hidden_layers', type=int, default=2, help='Hidden layers')
    parser.add_argument('--activation', type=str, default='softplus', help='Activation')
    parser.add_argument('--solver', type=str, default='adam', help='Optimizer')
    
    # Generative model arguments
    parser.add_argument('--loss_weight', type=float, default=1.0, help='Reconstruction loss weight')
    parser.add_argument('--sparsity_ratio', type=float, default=1.0, help='Sparsity ratio')
    parser.add_argument('--loss_type', type=str, default='KL', choices=['KL', 'CE'], help='Loss type')
    parser.add_argument('--temperature', type=float, default=3.0, help='Softmax temperature')
    
    # Wandb arguments
    parser.add_argument('--wandb_project', type=str, default=None, help='W&B project name')
    parser.add_argument('--wandb_offline', action='store_true', help='Offline mode')
    parser.add_argument('--load_run_id_or_name', type=str, default=None,
                        help='Load from previous W&B run for re-evaluation')
    
    args = parser.parse_args()
    
    if args.load_run_id_or_name:
        run_reevaluate(args)
    else:
        if args.data_path is None:
            parser.error("the following arguments are required: --data_path")
        run(args)
