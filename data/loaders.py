"""Dataset loading utilities for topic modeling."""

import os
import json
import shutil
from typing import Optional
from datasets import (
    load_dataset,
    load_from_disk,
    get_dataset_config_names,
    concatenate_datasets,
    Dataset,
)
from huggingface_hub import hf_hub_download

# Directory for storing processed datasets locally
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "processed_data")


def get_hf_dataset(dataset_name: str, split: Optional[str] = None) -> Dataset:
    """Load a dataset from HuggingFace Hub.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        split: Dataset split to load (None or 'all' loads all splits)
        
    Returns:
        Concatenated dataset from all specified splits
    """
    configs = get_dataset_config_names(dataset_name, trust_remote_code=True)
    if 'default' in configs:
        configs = ['default']

    if len(configs) == 0:
        raise ValueError(f"No dataset configs found for {dataset_name}")

    datasets = []
    for cfg in configs:
        dataset = load_dataset(dataset_name, cfg, trust_remote_code=True)
        if split is None or split == 'all':
            all_splits = list(dataset.keys())
        else:
            all_splits = [split]

        for s in all_splits:
            datasets.append(dataset[s])

    dataset = concatenate_datasets(datasets)
    return dataset


def get_local_dataset(dataset_path: str) -> Dataset:
    """Load a dataset from a local TSV file.
    
    Args:
        dataset_path: Path to the TSV file
        
    Returns:
        Dataset loaded from the TSV file
    """
    if not os.path.basename(dataset_path).endswith('.tsv'):
        raise ValueError(f"Dataset {dataset_path} is not a TSV file")
    dataset = load_dataset("csv", data_files=dataset_path, delimiter='\t')
    dataset = dataset['train']
    return dataset


def load_or_download_dataset(
    repo_id: str,
    local_name: Optional[str] = None,
    force_download: bool = False
) -> tuple[Dataset, list[str], dict]:
    """Load processed dataset from local cache or download from HuggingFace Hub.
    
    First tries to load from local processed_data/ directory. If not found,
    downloads from HuggingFace Hub and caches locally.
    
    Args:
        repo_id: HuggingFace repository ID (e.g., 'username/dataset-name')
        local_name: Local directory name (defaults to repo_id basename)
        force_download: Force re-download even if local exists
        
    Returns:
        Tuple of (dataset, vocab, metadata)
    """
    if local_name is None:
        local_name = repo_id.split("/")[-1]
    
    local_path = os.path.join(PROCESSED_DATA_DIR, local_name)
    
    # Try to load from local first
    if os.path.exists(local_path) and not force_download:
        print(f"Loading dataset from local cache: {local_path}")
        dataset = load_from_disk(local_path)
        
        vocab_file = os.path.join(local_path, "vocab.json")
        metadata_file = os.path.join(local_path, "metadata.json")
        
        if os.path.exists(vocab_file) and os.path.exists(metadata_file):
            with open(vocab_file, 'r') as f:
                vocab = json.load(f)
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            return dataset, vocab, metadata
        else:
            print(f"Warning: vocab.json or metadata.json not found in {local_path}")
    
    # Download from HuggingFace Hub
    print(f"Downloading dataset from HuggingFace Hub: {repo_id}")
    dataset = load_dataset(repo_id, split='train')
    
    # Download vocab and metadata
    vocab_hf_path = hf_hub_download(repo_id=repo_id, filename='vocab.json', repo_type='dataset')
    metadata_hf_path = hf_hub_download(repo_id=repo_id, filename='metadata.json', repo_type='dataset')
    
    with open(vocab_hf_path, 'r') as f:
        vocab = json.load(f)
    with open(metadata_hf_path, 'r') as f:
        metadata = json.load(f)
    
    # Save to local cache
    os.makedirs(local_path, exist_ok=True)
    dataset.save_to_disk(local_path)
    
    # Copy vocab and metadata to local path
    shutil.copy(vocab_hf_path, os.path.join(local_path, "vocab.json"))
    shutil.copy(metadata_hf_path, os.path.join(local_path, "metadata.json"))
    
    print(f"Dataset cached locally at: {local_path}")
    
    return dataset, vocab, metadata


def save_processed_dataset(
    dataset: Dataset,
    vocab: list[str],
    metadata: dict,
    local_name: str
) -> str:
    """Save processed dataset to local processed_data/ directory.
    
    Args:
        dataset: HuggingFace Dataset to save
        vocab: List of vocabulary words
        metadata: Metadata dictionary
        local_name: Local directory name
        
    Returns:
        Path to the saved dataset
    """
    local_path = os.path.join(PROCESSED_DATA_DIR, local_name)
    os.makedirs(local_path, exist_ok=True)
    
    # Save dataset
    dataset.save_to_disk(local_path)
    
    # Save vocab
    vocab_path = os.path.join(local_path, "vocab.json")
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=2)
    
    # Save metadata
    metadata_path = os.path.join(local_path, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dataset saved to: {local_path}")
    return local_path


def save_and_upload_dataset(
    dataset: Dataset,
    vocab: list[str],
    metadata: dict,
    local_name: str,
    hf_repo_name: Optional[str] = None,
    private: bool = False
) -> str:
    """Save processed dataset locally and optionally upload to HuggingFace Hub.
    
    Args:
        dataset: HuggingFace Dataset to save
        vocab: List of vocabulary words
        metadata: Metadata dictionary
        local_name: Local directory name
        hf_repo_name: HuggingFace repository ID (e.g., 'username/dataset-name')
        private: Whether to make the HF dataset private (default: False for public)
        
    Returns:
        Path to the saved local dataset
    """
    from huggingface_hub import HfApi
    
    # Save locally first
    local_path = save_processed_dataset(dataset, vocab, metadata, local_name)
    
    # Upload to HuggingFace Hub if repo name provided
    if hf_repo_name:
        print(f"\nUploading dataset to HuggingFace Hub: {hf_repo_name}")
        
        try:
            # Push dataset
            dataset.push_to_hub(
                hf_repo_name,
                private=private,
            )
            print(f"  ✓ Dataset pushed to hub")
            
            # Upload additional files
            vocab_path = os.path.join(local_path, "vocab.json")
            metadata_path = os.path.join(local_path, "metadata.json")
            
            api = HfApi()
            
            api.upload_file(
                path_or_fileobj=vocab_path,
                path_in_repo="vocab.json",
                repo_id=hf_repo_name,
                repo_type="dataset",
            )
            print(f"  ✓ vocab.json uploaded ({len(vocab)} words)")
            
            api.upload_file(
                path_or_fileobj=metadata_path,
                path_in_repo="metadata.json",
                repo_id=hf_repo_name,
                repo_type="dataset",
            )
            print(f"  ✓ metadata.json uploaded")
            
            print(f"\n✓ Dataset successfully uploaded to: https://huggingface.co/datasets/{hf_repo_name}")
            
        except Exception as e:
            print(f"\n✗ Failed to upload to HuggingFace Hub: {e}")
            print(f"  Dataset is still saved locally at: {local_path}")
    
    return local_path


def load_processed_dataset(repo_id: str) -> tuple[Dataset, list[str], dict]:
    """Load a processed dataset from HuggingFace Hub (deprecated, use load_or_download_dataset).
    
    Args:
        repo_id: HuggingFace repository ID (e.g., 'username/dataset-name')
        
    Returns:
        Tuple of (dataset, vocab, metadata)
    """
    # Redirect to load_or_download_dataset for backward compatibility
    return load_or_download_dataset(repo_id)


def load_bow(dataset: Dataset) -> list[str]:
    """Load bag-of-words representations from a dataset.
    
    Args:
        dataset: HuggingFace Dataset with 'bow' column
        
    Returns:
        List of bow strings
    """
    if 'bow' not in dataset.column_names:
        raise ValueError("Dataset does not have 'bow' column")
    return dataset['bow']


def load_labels(dataset: Dataset) -> list:
    """Load labels from a dataset.
    
    Args:
        dataset: HuggingFace Dataset with 'label' column
        
    Returns:
        List of labels
    """
    if 'label' not in dataset.column_names:
        raise ValueError("Dataset does not have 'label' column")
    return dataset['label']


def load_vocab_from_hub(repo_id: str) -> list[str]:
    """Load vocabulary from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repository ID
        
    Returns:
        List of vocabulary words
    """
    vocab_path = hf_hub_download(repo_id=repo_id, filename='vocab.json', repo_type='dataset')
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    return vocab


def load_metadata_from_hub(repo_id: str) -> dict:
    """Load metadata from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repository ID
        
    Returns:
        Metadata dictionary
    """
    metadata_path = hf_hub_download(repo_id=repo_id, filename='metadata.json', repo_type='dataset')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata

