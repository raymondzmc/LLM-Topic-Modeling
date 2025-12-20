import numpy as np
from models.octis.contextualized_topic_models.datasets import CTMDataset
from typing import Any, Union
from tqdm import tqdm
from datasets import Dataset


def get_ctm_dataset_from_processed_data(data: Union[dict[str, Any], Dataset], vocab: list[str], layer_idx: int = -1) -> CTMDataset:
    """Create CTM dataset from processed generative model data.
    
    Args:
        data: Dictionary with 'input_embeddings' and 'next_word_logits' OR HuggingFace Dataset
        vocab: Vocabulary list
        layer_idx: Which layer's embeddings to use (-1 for last layer)
        
    Returns:
        CTMDataset ready for training and inference
    """
    # Check if data is a HuggingFace Dataset or a dict containing one
    dataset = None
    if isinstance(data, Dataset):
        dataset = data
    elif isinstance(data, dict) and 'hf_dataset' in data:
        dataset = data['hf_dataset']

    if dataset is not None:
        # Optimized loading from HuggingFace dataset with progress bar
        n_samples = len(dataset)
        
        # Peek at first element to determine shapes
        first_item = dataset[0]
        first_emb = np.array(first_item['input_embeddings'])
        first_logits = np.array(first_item['next_word_logits'])
        
        # Determine embedding shape
        if first_emb.ndim == 2:
            emb_dim = first_emb.shape[1]
        else:
            emb_dim = first_emb.shape[0]
            
        logits_dim = first_logits.shape[0]
        
        # Pre-allocate arrays
        x_embeddings = np.zeros((n_samples, emb_dim), dtype=np.float32)
        y = np.zeros((n_samples, logits_dim), dtype=np.float32)
        
        print(f"Loading {n_samples} embeddings and logits...")
        for i, item in enumerate(tqdm(dataset, desc="Extracting features")):
            # Process embeddings
            emb = np.array(item['input_embeddings'])
            if emb.ndim == 2:
                x_embeddings[i] = emb[layer_idx]
            else:
                x_embeddings[i] = emb
            
            # Process logits
            y[i] = np.array(item['next_word_logits'])
            
    else:
        # Legacy path: loading from pre-loaded lists
        embeddings = data['input_embeddings']
        
        # Handle multi-layer embeddings (shape: n_docs x n_layers x hidden_dim)
        first_embedding = np.array(embeddings[0])
        if first_embedding.ndim == 2:
            # Multi-layer embeddings: select specific layer
            x_embeddings = np.stack([np.array(emb)[layer_idx] for emb in embeddings])
        else:
            # Single layer embedding
            x_embeddings = np.stack(embeddings)

        y = np.stack(data['next_word_logits'])
    
    # Create dummy x_bow (not used by GenerativeTM but required for CTMDataset compatibility)
    idx2token = {i: token for i, token in enumerate(vocab)}
    dataset = CTMDataset(
        x_bow=None,
        x_embeddings=x_embeddings,
        idx2token=idx2token,
        y=y,
    )
    return dataset