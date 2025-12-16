from collections import Counter
from datasets import Dataset
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_list
from scipy.sparse import lil_matrix
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import numpy as np
from contextualized_topic_models.datasets.dataset import CTMDataset
from typing import Any


def get_ctm_dataset(data: dict[str, Any], vocab: list[str]):
    vocab2id = {w: i for i, w in enumerate(vocab)}
    vec = CountVectorizer(vocabulary=vocab2id, token_pattern=r'(?u)\b[\w+|\-]+\b')
    corpus = [' '.join(x) for x in data['words']]
    vec.fit(corpus)
    X_bow = vec.transform(corpus)
    X_contextual = bert_embeddings_from_list(
        corpus,
        sbert_model_to_load='paraphrase-multilingual-mpnet-base-v2',
    )
    idx2token = {v: k for (k, v) in vec.vocabulary_.items()}
    dataset = CTMDataset(X_contextual=X_contextual,
                         X_bow=X_bow,
                         idx2token=idx2token)
    return dataset


def get_ctm_dataset_generative(data: dict[str, Any], vocab: list[str], layer_idx: int = -1):
    """Create CTM dataset from processed generative model data.
    
    Args:
        data: Dictionary with 'input_embeddings' and 'next_word_logits'
        vocab: Vocabulary list
        layer_idx: Which layer's embeddings to use (-1 for last layer)
        
    Returns:
        CTMDataset ready for training
    """
    embeddings = data['input_embeddings']
    
    # Handle multi-layer embeddings (shape: n_docs x n_layers x hidden_dim)
    first_embedding = np.array(embeddings[0])
    if first_embedding.ndim == 2:
        # Multi-layer embeddings: select specific layer
        X_contextual = np.stack([np.array(emb)[layer_idx] for emb in embeddings])
    else:
        # Single layer embedding
        X_contextual = np.stack(embeddings)
    
    X_bow = np.stack(data['next_word_logits'])
    idx2token = {i: token for i, token in enumerate(vocab)}
    dataset = CTMDataset(X_contextual=X_contextual,
                         X_bow=X_bow,
                         idx2token=idx2token)
    return dataset