"""Add missing cv_wiki metric to wandb runs.

This script:
1. Fetches runs missing cv_wiki from wandb
2. Downloads the model artifact (contains model_output.pt with topics)
3. Computes cv_wiki using Palmetto (locally available)
4. Loads existing evaluation results and adds cv_wiki
5. Creates a new wandb run with the same name and uploads updated results
"""

import os
import sys
import json
import tempfile
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import wandb
import numpy as np
from collections import defaultdict

from evaluation.coherence_metrics import PalmettoCoherence
from settings import settings


# Run IDs missing cv_wiki (identified from diagnostic)
MISSING_RUNS = {
    "tweet_topic": [
        "1vsxjr6p",  # K25_temp0.5
        "im513ly4",  # K25_temp1.0
        "fk3m583c",  # K25_temp2.0
        "2fz4383y",  # K25_temp6.0
        "8ryy6nnn",  # K25_temp7.0
        "n8svjg1f",  # K25_temp9.0
        "p4ujaxlb",  # K50_temp0.5
        "w0pi4ib3",  # K50_temp1.0
        "qjitjeue",  # K50_temp2.0
        "gnyvospq",  # K50_temp6.0
        "rfo1v4lr",  # K50_temp7.0
        "tvwglowj",  # K50_temp9.0
        "58tyn9k3",  # K75_temp0.5
        "l61lq3gm",  # K75_temp1.0
        "p4o6ngvf",  # K75_temp2.0
        "oorxhlza",  # K75_temp6.0
        "of0d26cz",  # K75_temp7.0
        "maplu78e",  # K75_temp9.0
        "lvfgywl2",  # K100_temp0.5
        "ylgtx32l",  # K100_temp1.0
        "r4rsd23l",  # K100_temp2.0
        "ptuu4xlw",  # K100_temp6.0
        "v65ubuw1",  # K100_temp7.0
        "5n9cr88r",  # K100_temp9.0
    ],
    "stackoverflow": [
        "zcbncbi7",  # K25_temp0.5
        "urgzf3zt",  # K25_temp1.0
        "duo9ynrz",  # K25_temp2.0
        "v6nh48ut",  # K25_temp6.0
        "u1pyr1lt",  # K25_temp7.0
        "51dr8zji",  # K25_temp9.0
        "qwobax3b",  # K50_temp0.5
        "iwdibac9",  # K50_temp1.0
        "h91xlehh",  # K50_temp2.0
        "wwupp9q7",  # K50_temp6.0
        "edmg08zl",  # K50_temp7.0
        "y0es1qz4",  # K50_temp9.0
        "9uf3cp89",  # K75_temp0.5
        "imrzn2zl",  # K75_temp1.0
        "4juzdwa5",  # K75_temp2.0
        "n8yormin",  # K75_temp4.0
        "7esh34oo",  # K75_temp6.0
        "2sn2hg9v",  # K75_temp7.0
        "31hli8nc",  # K75_temp9.0
        "rsk7xfw5",  # K100_temp0.5
        "x1fj8lw5",  # K100_temp1.0
        "589d0uhp",  # K100_temp2.0
        "naz0jtl9",  # K100_temp4.0
        "pgg9izue",  # K100_temp5.0
        "ec65zgeq",  # K100_temp6.0
        "mfrn2hqa",  # K100_temp7.0
        "0omahult",  # K100_temp8.0
        "ucbr9dqq",  # K100_temp9.0
    ],
}


def get_palmetto_coherence(top_words: int = 10) -> PalmettoCoherence:
    """Initialize Palmetto coherence metric."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    palmetto_jar = os.path.join(project_root, "data/wikipedia/palmetto-0.1.0-jar-with-dependencies.jar")
    wikipedia_index = os.path.join(project_root, "data/wikipedia/wikipedia_bd")
    
    if not os.path.exists(palmetto_jar):
        raise FileNotFoundError(f"Palmetto JAR not found: {palmetto_jar}")
    if not os.path.exists(wikipedia_index):
        raise FileNotFoundError(f"Wikipedia index not found: {wikipedia_index}")
    
    return PalmettoCoherence(
        palmetto_jar=palmetto_jar,
        wikipedia_index=wikipedia_index,
        measure="C_V",
        topk=top_words,
    )


def compute_aggregate_results(results_dir: str) -> dict:
    """Compute aggregated results from per-seed results."""
    aggregated_results = defaultdict(float)
    counts = defaultdict(int)
    
    for seed_dir in os.listdir(results_dir):
        results_file = os.path.join(results_dir, seed_dir, 'evaluation_results.json')
        if os.path.exists(results_file):
            with open(results_file, encoding='utf-8') as f:
                results = json.load(f)
            
            if isinstance(results, list):
                results = results[1]
            
            for k, v in results.items():
                aggregated_results[k] += v
                counts[k] += 1
    
    for k in aggregated_results:
        aggregated_results[k] /= counts[k]
    
    return dict(aggregated_results)


def add_cv_wiki_to_run(run_id: str, project: str, dry_run: bool = False, top_words: int = 10, use_dummy: bool = False, force: bool = False):
    """Add cv_wiki metric to a single wandb run."""
    api = wandb.Api()
    
    # Get source run
    source_run = api.run(f"{settings.wandb_entity}/{project}/{run_id}")
    print(f"\nProcessing: {source_run.name} ({run_id})")
    
    # Check if already has cv_wiki
    existing_cv_wiki = source_run.summary.get("avg/cv_wiki")
    if existing_cv_wiki is not None and not force:
        print(f"  Already has cv_wiki: {existing_cv_wiki:.4f}, skipping (use --force to re-compute)")
        return
    elif existing_cv_wiki is not None and force:
        print(f"  Existing cv_wiki: {existing_cv_wiki:.4f}, re-computing (--force)")
    
    # Get model artifact
    artifacts = list(source_run.logged_artifacts())
    model_artifacts = [a for a in artifacts if a.type == "model"]
    
    if len(model_artifacts) == 0:
        print(f"  No model artifacts found, skipping")
        return
    
    artifact = model_artifacts[-1]
    print(f"  Artifact: {artifact.name} (v{artifact.version})")
    
    # Initialize Palmetto (or use dummy)
    palmetto = None
    if not use_dummy:
        palmetto = get_palmetto_coherence(top_words=top_words)
    else:
        print("  Using DUMMY cv_wiki values for testing")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download artifact
        artifact_dir = artifact.download(root=temp_dir)
        print(f"  Downloaded to: {artifact_dir}")
        
        # Get metadata
        metadata = artifact.metadata or {}
        num_seeds = metadata.get('num_seeds', 1)
        model_name = metadata.get('model', 'unknown')
        dataset_name = metadata.get('dataset', 'unknown')
        num_topics = metadata.get('num_topics', 0)
        has_labels = metadata.get('has_labels', False)
        
        # Process each seed
        results_dir = os.path.join(temp_dir, 'updated')
        os.makedirs(results_dir, exist_ok=True)
        
        for seed in range(num_seeds):
            seed_dir = os.path.join(artifact_dir, f"seed_{seed}")
            new_seed_dir = os.path.join(results_dir, f"seed_{seed}")
            os.makedirs(new_seed_dir, exist_ok=True)
            
            # Load model output to get topics
            model_output_path = os.path.join(seed_dir, 'model_output.pt')
            if not os.path.exists(model_output_path):
                print(f"  [Seed {seed}] model_output.pt not found, skipping")
                continue
            
            model_output = torch.load(model_output_path, weights_only=False)
            topics = model_output.get('topics', [])
            
            if not topics:
                print(f"  [Seed {seed}] No topics found, skipping")
                continue
            
            # Compute cv_wiki
            if use_dummy:
                cv_wiki_score = 0.999  # Dummy value for testing
            else:
                cv_wiki_score = palmetto.score({'topics': topics})
            print(f"  [Seed {seed}] cv_wiki: {cv_wiki_score:.4f}{'  [DUMMY]' if use_dummy else ''}")
            
            # Load existing evaluation results
            eval_results_path = os.path.join(seed_dir, 'evaluation_results.json')
            if os.path.exists(eval_results_path):
                with open(eval_results_path, encoding='utf-8') as f:
                    evaluation_results = json.load(f)
            else:
                evaluation_results = {}
            
            # Add cv_wiki
            evaluation_results['cv_wiki'] = float(cv_wiki_score)
            
            # Save updated evaluation results
            with open(os.path.join(new_seed_dir, 'evaluation_results.json'), 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f)
            
            # Copy model output and topics
            torch.save(model_output, os.path.join(new_seed_dir, 'model_output.pt'))
            
            topics_path = os.path.join(seed_dir, 'topics.json')
            if os.path.exists(topics_path):
                with open(topics_path, encoding='utf-8') as f:
                    topics_data = json.load(f)
                with open(os.path.join(new_seed_dir, 'topics.json'), 'w', encoding='utf-8') as f:
                    json.dump(topics_data, f)
        
        # Compute aggregated results
        averaged_results = compute_aggregate_results(results_dir)
        print(f"  Averaged cv_wiki: {averaged_results.get('cv_wiki', 'N/A')}")
        
        with open(os.path.join(results_dir, 'averaged_results.json'), 'w', encoding='utf-8') as f:
            json.dump(averaged_results, f)
        
        # Copy labels if they exist
        labels_path = os.path.join(artifact_dir, 'labels.json')
        if os.path.exists(labels_path):
            with open(labels_path, encoding='utf-8') as f:
                labels = json.load(f)
            with open(os.path.join(results_dir, 'labels.json'), 'w', encoding='utf-8') as f:
                json.dump(labels, f)
        
        # Copy vocab if it exists
        vocab_path = os.path.join(artifact_dir, 'vocab.json')
        if os.path.exists(vocab_path):
            with open(vocab_path, encoding='utf-8') as f:
                vocab = json.load(f)
            with open(os.path.join(results_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
                json.dump(vocab, f)
        
        # Copy config if it exists
        config_path = os.path.join(artifact_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, encoding='utf-8') as f:
                config = json.load(f)
            with open(os.path.join(results_dir, 'config.json'), 'w', encoding='utf-8') as f:
                json.dump(config, f)
        
        if dry_run:
            print(f"  [DRY RUN] Would upload to wandb with name: {source_run.name}")
            return
        
        # Create new wandb run with same name
        new_run = wandb.init(
            project=project,
            entity=settings.wandb_entity,
            name=source_run.name,
            config={
                "source_run_id": source_run.id,
                "source_run_name": source_run.name,
                "model": model_name,
                "dataset": dataset_name,
                "num_topics": num_topics,
                "num_seeds": num_seeds,
                "top_words": top_words,
                "cv_wiki_added": True,
            },
        )
        
        # Log per-seed results
        for seed in range(num_seeds):
            seed_results_path = os.path.join(results_dir, f"seed_{seed}", 'evaluation_results.json')
            if os.path.exists(seed_results_path):
                with open(seed_results_path, encoding='utf-8') as f:
                    seed_results = json.load(f)
                new_run.log({f"seed_{seed}/{k}": v for k, v in seed_results.items()})
        
        # Log averaged results
        new_run.log({f"avg/{k}": v for k, v in averaged_results.items()})
        
        # Upload artifact
        new_artifact = wandb.Artifact(
            name=f"{model_name}-K{num_topics}-{dataset_name}",
            type="model",
            description=f"Added cv_wiki from {source_run.name} ({source_run.id})",
            metadata={
                "model": model_name,
                "dataset": dataset_name,
                "num_topics": num_topics,
                "num_seeds": num_seeds,
                "top_words": top_words,
                "has_labels": has_labels,
                "source_run_id": source_run.id,
                "cv_wiki_added": True,
            }
        )
        new_artifact.add_dir(results_dir)
        new_run.log_artifact(new_artifact)
        new_run.finish()
        
        print(f"  Uploaded to wandb: {source_run.name}")


def main():
    parser = argparse.ArgumentParser(description="Add missing cv_wiki metric to wandb runs")
    parser.add_argument('--run_id', type=str, default=None, help='Specific run ID to process')
    parser.add_argument('--project', type=str, default=None, help='Project name (required with --run_id)')
    parser.add_argument('--dry_run', action='store_true', help='Dry run (no upload)')
    parser.add_argument('--top_words', type=int, default=10, help='Top words for coherence')
    parser.add_argument('--all', action='store_true', help='Process all missing runs')
    parser.add_argument('--dummy', action='store_true', help='Use dummy cv_wiki value (for testing)')
    parser.add_argument('--force', action='store_true', help='Re-compute cv_wiki even if it already exists')
    args = parser.parse_args()
    
    if args.run_id:
        if not args.project:
            parser.error("--project is required when using --run_id")
        add_cv_wiki_to_run(args.run_id, args.project, dry_run=args.dry_run, top_words=args.top_words, use_dummy=args.dummy, force=args.force)
    elif args.all:
        print("=" * 60)
        print("Adding cv_wiki to all missing runs")
        print("=" * 60)
        
        for project, run_ids in MISSING_RUNS.items():
            print(f"\n{'='*60}")
            print(f"Project: {project} ({len(run_ids)} runs)")
            print("=" * 60)
            
            for run_id in run_ids:
                try:
                    add_cv_wiki_to_run(run_id, project, dry_run=args.dry_run, top_words=args.top_words, use_dummy=args.dummy, force=args.force)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    continue
        
        print("\n" + "=" * 60)
        print("Done!")
        print("=" * 60)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Process single run:")
        print("  python scripts/add_missing_cv_wiki.py --run_id 1vsxjr6p --project tweet_topic")
        print("")
        print("  # Dry run (no upload):")
        print("  python scripts/add_missing_cv_wiki.py --run_id 1vsxjr6p --project tweet_topic --dry_run")
        print("")
        print("  # Process all missing runs:")
        print("  python scripts/add_missing_cv_wiki.py --all")


if __name__ == "__main__":
    main()

