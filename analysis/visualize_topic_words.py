"""Visualize topic words for each dataset by class alignment.

For each dataset, finds the topic most aligned with each ground-truth class
(based on purity/contingency matrix), and displays the top-15 topic words
for that topic across all methods.

Output: Markdown tables showing topic words per class for each method.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import tempfile
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import wandb
from sklearn.metrics.cluster import contingency_matrix

from settings import settings

# Configuration
DATASETS = ["20_newsgroups", "tweet_topic", "stackoverflow"]
K_VALUE = 50
REQUIRED_NUM_SEEDS = 5
TOP_WORDS = 15

# Baseline methods
BASELINE_METHODS = ["lda", "prodlda", "zeroshot", "combined", "etm", "bertopic", "ecrtm", "fastopic"]

# Generative LLM models
GENERATIVE_LLM_MODELS = [
    "ERNIE-4.5-0.3B-PT",
    "Llama-3.1-8B-Instruct",
    "Llama-3.2-1B-Instruct",
]

# Display names for methods
METHOD_DISPLAY_NAMES = {
    "lda": "LDA",
    "prodlda": "ProdLDA",
    "zeroshot": "ZeroShotTM",
    "combined": "CombinedTM",
    "etm": "ETM",
    "bertopic": "BERTopic",
    "ecrtm": "ECRTM",
    "fastopic": "FASTopic",
    "generative_ERNIE-4.5-0.3B-PT": "ERNIE",
    "generative_Llama-3.1-8B-Instruct": "Llama-8B",
    "generative_Llama-3.2-1B-Instruct": "Llama-1B",
}

# Class label names for each dataset
# 20_newsgroups: Standard newsgroup names
NEWSGROUPS_CLASSES = [
    "alt.atheism", "comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware", "comp.windows.x", "misc.forsale", "rec.autos",
    "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey", "sci.crypt",
    "sci.electronics", "sci.med", "sci.space", "soc.religion.christian",
    "talk.politics.guns", "talk.politics.mideast", "talk.politics.misc", "talk.religion.misc"
]

# tweet_topic: Label names from dataset
TWEET_TOPIC_CLASSES = {
    0: "arts_&_culture",
    1: "business_&_entrepreneurs",
    2: "celebrity_&_pop_culture",
    3: "diaries_&_daily_life",
    4: "family",
    5: "fashion_&_style",
    6: "film_tv_&_video",
    7: "fitness_&_health",
    8: "food_&_dining",
    9: "gaming",
    10: "learning_&_educational",
    11: "music",
    12: "news_&_social_concern",
    13: "other_hobbies",
    14: "relationships",
    15: "science_&_technology",
    16: "sports",
    17: "travel_&_adventure",
    18: "youth_&_student_life",
}

# stackoverflow: Class names from the 20 programming categories
STACKOVERFLOW_CLASSES = {
    0: "ajax",
    1: "apache",
    2: "bash",
    3: "svn",
    4: "drupal",
    5: "excel",
    6: "haskell",
    7: "visual-studio",
    8: "spring",
    9: "osx",
    10: "linux",
    11: "magento",
    12: "matlab",
    13: "oracle",
    14: "scala",
    15: "asp.net",
    16: "sharepoint",
    17: "wordpress",
    18: "linq",
    19: "cocoa",
}


def get_method_key_from_run(run) -> Optional[str]:
    """Extract method key from a wandb run.
    
    For baseline methods, returns config.model only if run name exactly matches {model}_K{num}.
    For generative methods, returns 'generative_{LLM_MODEL}' only if run name exactly matches.
    """
    model = run.config.get("model", "")
    run_name = run.name
    
    if model == "generative":
        for llm_model in GENERATIVE_LLM_MODELS:
            expected_prefix = f"generative_{llm_model}_K"
            if run_name.startswith(expected_prefix):
                suffix = run_name[len(expected_prefix):]
                if suffix.isdigit():
                    return f"generative_{llm_model}"
        return None
    elif model in BASELINE_METHODS:
        expected_prefix = f"{model}_K"
        if run_name.startswith(expected_prefix):
            suffix = run_name[len(expected_prefix):]
            if suffix.isdigit():
                return model
        return None
    else:
        return None


def fetch_k50_runs(dataset: str) -> dict:
    """Fetch K=50 runs for all methods from wandb.
    
    Returns:
        Dict mapping method_key to wandb run object
    """
    api = wandb.Api()
    project_path = f"{settings.wandb_entity}/{dataset}"
    
    print(f"\nFetching K={K_VALUE} runs from {project_path}...")
    
    try:
        runs = api.runs(
            project_path,
            filters={"state": "finished"},
            order="-created_at",
        )
        runs_list = list(runs)
        print(f"  Found {len(runs_list)} finished runs")
    except Exception as e:
        print(f"  Error fetching runs: {e}")
        return {}
    
    # Find the latest K=50 run for each method
    method_runs = {}
    seen_methods = set()
    
    for run in runs_list:
        method_key = get_method_key_from_run(run)
        if method_key is None:
            continue
        
        num_topics = run.config.get("num_topics")
        if num_topics != K_VALUE:
            continue
        
        num_seeds = run.config.get("num_seeds", 0)
        if num_seeds != REQUIRED_NUM_SEEDS:
            continue
        
        # Take only the first (latest) run for each method
        if method_key in seen_methods:
            continue
        seen_methods.add(method_key)
        
        method_runs[method_key] = run
        print(f"  Found: {method_key} -> {run.name}")
    
    return method_runs


def download_and_load_artifact(run, temp_dir: str) -> Optional[dict]:
    """Download artifact and load model output.
    
    Returns:
        Dict with keys: 'topics', 'topic_document_matrix', 'labels'
        or None if artifact not found or loading failed
    """
    # Find model artifact
    artifacts = list(run.logged_artifacts())
    model_artifacts = [a for a in artifacts if a.type == "model"]
    
    if len(model_artifacts) == 0:
        print(f"    No model artifacts found for {run.name}")
        return None
    
    artifact = model_artifacts[-1]
    artifact_dir = artifact.download(root=temp_dir)
    
    # Load data from seed_0 (or first available seed)
    seed_dir = os.path.join(artifact_dir, "seed_0")
    if not os.path.exists(seed_dir):
        # Try to find any seed directory
        for item in os.listdir(artifact_dir):
            if item.startswith("seed_"):
                seed_dir = os.path.join(artifact_dir, item)
                break
        else:
            print(f"    No seed directory found in artifact for {run.name}")
            return None
    
    # Load model output
    model_output_path = os.path.join(seed_dir, "model_output.pt")
    if not os.path.exists(model_output_path):
        print(f"    model_output.pt not found for {run.name}")
        return None
    
    try:
        model_output = torch.load(model_output_path, weights_only=False)
    except Exception as e:
        print(f"    Error loading model_output.pt: {e}")
        return None
    
    # Extract required data
    topics = model_output.get("topics", [])
    topic_doc_matrix = model_output.get("topic-document-matrix")
    
    if not topics:
        print(f"    No topics found in model output for {run.name}")
        return None
    
    if topic_doc_matrix is None:
        print(f"    No topic-document-matrix found for {run.name}")
        return None
    
    # Load labels
    labels_path = os.path.join(artifact_dir, "labels.json")
    if not os.path.exists(labels_path):
        print(f"    labels.json not found for {run.name}")
        return None
    
    with open(labels_path, encoding="utf-8") as f:
        labels = json.load(f)
    
    return {
        "topics": topics,
        "topic_document_matrix": topic_doc_matrix,
        "labels": labels,
    }


def compute_topic_class_alignment(topic_doc_matrix: np.ndarray, labels: list) -> dict:
    """Compute which topic is most aligned with each class.
    
    Uses contingency matrix where rows=classes, cols=topics.
    For each class, returns the topic with maximum documents from that class.
    
    Args:
        topic_doc_matrix: Shape (K, n_documents)
        labels: List of class labels for each document
        
    Returns:
        Dict mapping class_id to topic_id
    """
    labels_array = np.asarray(labels)
    
    # Hard-assign each document to its most probable topic
    topic_assignments = topic_doc_matrix.argmax(axis=0)
    
    # Build contingency matrix: rows=classes, cols=topics
    cmat = contingency_matrix(labels_array, topic_assignments)
    
    # For each class, find the topic with maximum count
    unique_classes = np.unique(labels_array)
    class_to_topic = {}
    
    for i, class_id in enumerate(unique_classes):
        best_topic = cmat[i].argmax()
        class_to_topic[int(class_id)] = int(best_topic)
    
    return class_to_topic


def get_class_names(dataset: str, class_ids: list) -> dict:
    """Get human-readable class names for each class ID.
    
    Args:
        dataset: Dataset name
        class_ids: List of class IDs found in the data
        
    Returns:
        Dict mapping class_id to class_name
    """
    if dataset == "20_newsgroups":
        # Class IDs are indices into NEWSGROUPS_CLASSES
        return {cid: NEWSGROUPS_CLASSES[cid] for cid in class_ids if cid < len(NEWSGROUPS_CLASSES)}
    elif dataset == "tweet_topic":
        return {cid: TWEET_TOPIC_CLASSES.get(cid, f"class_{cid}") for cid in class_ids}
    elif dataset == "stackoverflow":
        return {cid: STACKOVERFLOW_CLASSES.get(cid, f"class_{cid}") for cid in class_ids}
    else:
        return {cid: f"class_{cid}" for cid in class_ids}


def generate_markdown_table(
    dataset: str,
    method_data: dict,
    class_names: dict,
    method_order: list,
) -> str:
    """Generate markdown table for a dataset.
    
    Args:
        dataset: Dataset name
        method_data: Dict[method_key, Dict[class_id, list[str]]] - topic words per class per method
        class_names: Dict[class_id, class_name]
        method_order: Order of methods in table columns
        
    Returns:
        Markdown string
    """
    lines = []
    lines.append(f"## {dataset} (K={K_VALUE})")
    lines.append("")
    
    # Sort classes by name
    sorted_classes = sorted(class_names.items(), key=lambda x: x[1])
    
    # For each class, create a section
    for class_id, class_name in sorted_classes:
        lines.append(f"### {class_name}")
        lines.append("")
        
        # Header
        header = "| Method | Top-15 Topic Words |"
        separator = "|--------|-------------------|"
        lines.append(header)
        lines.append(separator)
        
        # Rows
        for method_key in method_order:
            display_name = METHOD_DISPLAY_NAMES.get(method_key, method_key)
            
            if method_key in method_data and class_id in method_data[method_key]:
                words = method_data[method_key][class_id][:TOP_WORDS]
                words_str = ", ".join(words)
            else:
                words_str = "-"
            
            lines.append(f"| {display_name} | {words_str} |")
        
        lines.append("")
    
    return "\n".join(lines)


def process_dataset(dataset: str) -> Optional[str]:
    """Process a single dataset and return markdown table.
    
    Returns:
        Markdown string or None if no data found
    """
    print(f"\n{'='*60}")
    print(f"Processing: {dataset}")
    print(f"{'='*60}")
    
    # Fetch runs
    method_runs = fetch_k50_runs(dataset)
    
    if not method_runs:
        print(f"  No K={K_VALUE} runs found for {dataset}")
        return None
    
    # Define method order
    all_methods = BASELINE_METHODS + [f"generative_{llm}" for llm in GENERATIVE_LLM_MODELS]
    
    # Download and process each method's data
    method_data = {}
    class_ids_union = set()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for method_key in all_methods:
            if method_key not in method_runs:
                print(f"  Skipping {method_key}: no run found")
                continue
            
            run = method_runs[method_key]
            print(f"\n  Processing {method_key} ({run.name})...")
            
            # Download and load artifact
            data = download_and_load_artifact(run, temp_dir)
            if data is None:
                continue
            
            topics = data["topics"]
            topic_doc_matrix = data["topic_document_matrix"]
            labels = data["labels"]
            
            # Convert to numpy if needed
            if hasattr(topic_doc_matrix, "numpy"):
                topic_doc_matrix = topic_doc_matrix.numpy()
            elif not isinstance(topic_doc_matrix, np.ndarray):
                topic_doc_matrix = np.array(topic_doc_matrix)
            
            # Compute class-topic alignment
            class_to_topic = compute_topic_class_alignment(topic_doc_matrix, labels)
            
            # Store topic words for each class
            method_data[method_key] = {}
            for class_id, topic_id in class_to_topic.items():
                if topic_id < len(topics):
                    method_data[method_key][class_id] = topics[topic_id]
                else:
                    method_data[method_key][class_id] = []
                class_ids_union.add(class_id)
            
            print(f"    Loaded {len(topics)} topics, {len(labels)} documents, {len(class_to_topic)} classes")
    
    if not method_data:
        print(f"  No data loaded for {dataset}")
        return None
    
    # Get class names
    class_names = get_class_names(dataset, sorted(class_ids_union))
    
    # Generate markdown
    markdown = generate_markdown_table(
        dataset=dataset,
        method_data=method_data,
        class_names=class_names,
        method_order=all_methods,
    )
    
    return markdown


def main():
    print("=" * 60)
    print("Topic Words Visualization by Class Alignment")
    print("=" * 60)
    print(f"Entity: {settings.wandb_entity}")
    print(f"Datasets: {DATASETS}")
    print(f"K value: {K_VALUE}")
    print(f"Top words: {TOP_WORDS}")
    print(f"Methods: {len(BASELINE_METHODS)} baselines + {len(GENERATIVE_LLM_MODELS)} generative")
    print("=" * 60)
    
    output_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(output_dir, exist_ok=True)
    
    for dataset in DATASETS:
        markdown = process_dataset(dataset)
        
        if markdown:
            output_path = os.path.join(output_dir, f"topic_words_{dataset}.md")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown)
            print(f"\nSaved: {output_path}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()

