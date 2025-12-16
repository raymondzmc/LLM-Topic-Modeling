"""Processing utilities for topic modeling dataset creation."""

import os
import platform
import torch
from typing import Optional
from llm import jinja_template_manager


def get_device_info(device: torch.device) -> dict:
    """Get device information including CPU and GPU details.
    
    Args:
        device: PyTorch device
        
    Returns:
        Dictionary with device information
    """
    device_info = {
        "device_type": str(device),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }
    
    # CPU info
    device_info["cpu_count"] = os.cpu_count()
    
    # GPU info if available
    if torch.cuda.is_available():
        device_info["cuda_available"] = True
        device_info["cuda_version"] = torch.version.cuda
        device_info["gpu_count"] = torch.cuda.device_count()
        if device.type == "cuda":
            gpu_idx = device.index if device.index is not None else 0
            device_info["gpu_name"] = torch.cuda.get_device_name(gpu_idx)
            device_info["gpu_memory_total"] = torch.cuda.get_device_properties(gpu_idx).total_memory
            device_info["gpu_memory_allocated"] = torch.cuda.memory_allocated(gpu_idx)
    else:
        device_info["cuda_available"] = False
    
    return device_info


def collate_fn(
    batch: list[dict],
    tokenizer,
    content_key: str,
    label_key: Optional[str],
    vocab_set: set[str],
    instruction_template: str,
    prompt_template: str
) -> dict:
    """Collate function for DataLoader that prepares batched inputs with left padding.
    
    Args:
        batch: List of examples from dataset
        tokenizer: HuggingFace tokenizer
        content_key: Key for text content
        label_key: Key for labels (optional)
        vocab_set: Set of vocabulary words
        instruction_template: Jinja template for instruction
        prompt_template: Jinja template for prompt
        
    Returns:
        Dictionary with batched tensors and metadata
    """
    instruction = jinja_template_manager.render(instruction_template)
    contexts = []
    example_ids = []
    labels = []
    bow_lines = []
    
    for i, example in enumerate(batch):
        context = jinja_template_manager.render(
            prompt_template, 
            instruction=instruction, 
            document=example[content_key]
        )
        contexts.append(context.rstrip())
        example_ids.append(example['id'] if 'id' in example else example.get('idx', i))
        
        # Include label if label_key is specified
        if label_key is not None and label_key in example:
            labels.append(example[label_key])
        else:
            labels.append(None)
        
        # Create bow_line from words
        if 'words' in example:
            filtered_words = [w for w in example['words'] if w in vocab_set]
            bow_lines.append(" ".join(filtered_words))
        else:
            bow_lines.append("")
    
    # Tokenize with left padding
    encoded = tokenizer(
        contexts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        return_attention_mask=True
    )
    
    return {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
        'contexts': contexts,
        'ids': example_ids,
        'labels': labels,
        'bow_lines': bow_lines
    }


def extract_embeddings(
    hidden_states: tuple,
    attention_mask: torch.Tensor,
    batch_idx: int,
    hidden_state_layer: Optional[int],
    embedding_method: str
) -> list:
    """Extract embeddings from model hidden states.
    
    Args:
        hidden_states: Tuple of hidden states from model
        attention_mask: Attention mask tensor
        batch_idx: Index in batch
        hidden_state_layer: Layer to extract from (None for all layers)
        embedding_method: 'mean' or 'last'
        
    Returns:
        Embeddings as list (or list of lists for all layers)
    """
    if hidden_state_layer is not None:
        if embedding_method == 'mean':
            mask = attention_mask[batch_idx].unsqueeze(-1)
            hidden = hidden_states[hidden_state_layer][batch_idx]
            embeddings = (hidden * mask).sum(dim=0) / mask.sum()
            return embeddings.cpu().tolist()
        elif embedding_method == 'last':
            return hidden_states[hidden_state_layer][batch_idx, -1].cpu().tolist()
        else:
            raise ValueError(f"Unsupported embedding method: {embedding_method}")
    else:
        # Save hidden states from all layers
        if embedding_method == 'mean':
            mask = attention_mask[batch_idx].unsqueeze(-1)
            embeddings = []
            for h in hidden_states:
                hidden = h[batch_idx]
                layer_emb = (hidden * mask).sum(dim=0) / mask.sum()
                embeddings.append(layer_emb.cpu().tolist())
            return embeddings
        elif embedding_method == 'last':
            return [h[batch_idx, -1].cpu().tolist() for h in hidden_states]
        else:
            raise ValueError(f"Unsupported embedding method: {embedding_method}")

