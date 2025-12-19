import numpy as np
from models.octis.contextualized_topic_models.datasets import CTMDataset
from typing import Any


def get_ctm_dataset_from_processed_data(data: dict[str, Any], vocab: list[str], layer_idx: int = -1) -> CTMDataset:
    """Create CTM dataset from processed generative model data.
    
    Args:
        data: Dictionary with 'input_embeddings' and 'next_word_logits'
        vocab: Vocabulary list
        layer_idx: Which layer's embeddings to use (-1 for last layer)
        
    Returns:
        CTMDataset ready for training and inference
    """
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