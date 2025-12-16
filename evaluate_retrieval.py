import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
# scipy.special.kl_div and numpy.ma are no longer needed with batched KL.


# The old compute_kl_divergence function is removed as we use a batched version.

def compute_pairwise_kl_divergence_torch(P, device):
    """
    Compute pairwise KL divergence between rows of a matrix P using PyTorch.
    KL(P_i || P_j) = sum_k P_i[k] * (log P_i[k] - log P_j[k])
    This version computes the matrix row-by-row to be more memory-efficient.
    Args:
        P (torch.Tensor): Input tensor of shape (n_docs, n_features) where rows are distributions.
        device (torch.device): The device (CPU or CUDA) to perform calculations on.
    Returns:
        torch.Tensor: Pairwise KL divergence matrix of shape (n_docs, n_docs).
    """
    P = P.to(device)
    n_docs, n_features = P.shape
    P_norm = P / P.sum(dim=1, keepdims=True)
    P_stable = P_norm + 1e-12
    logP_all = P_stable.log()
    kl_matrix = torch.zeros((n_docs, n_docs))

    print(f"  Calculating KL divergence row-by-row for {n_docs} documents...")
    for i in tqdm(range(n_docs), desc="  KL Div Row", unit="doc", leave=False, dynamic_ncols=True):
        log_diff = logP_all[i, :].unsqueeze(0) - logP_all
        product = P_norm[i, :].unsqueeze(0) * log_diff
        kl_matrix[i, :] = product.sum(dim=-1).cpu()
    kl_matrix.fill_diagonal_(0)
    return kl_matrix


def compute_pairwise_cosine_similarity_torch(X, device):
    """
    Compute pairwise cosine similarity between rows of a matrix X using PyTorch.
    cosine_similarity(X_i, X_j) = (X_i·X_j) / (||X_i|| * ||X_j||)
    This version computes the matrix row-by-row to be more memory-efficient.
    Args:
        X (torch.Tensor): Input tensor of shape (n_docs, n_features) where rows are vectors.
        device (torch.device): The device (CPU or CUDA) to perform calculations on.
    Returns:
        torch.Tensor: Pairwise cosine similarity matrix of shape (n_docs, n_docs).
    """
    X = X.to(device)
    n_docs, n_features = X.shape
    # Normalize rows to unit length
    X_norm = torch.nn.functional.normalize(X, p=2, dim=1)
    sim_matrix = torch.zeros((n_docs, n_docs), device=device)
    
    print(f"  Calculating cosine similarity row-by-row for {n_docs} documents...")
    for i in tqdm(range(n_docs), desc="  Cosine Sim Row", unit="doc", leave=False, dynamic_ncols=True):
        # Dot product of normalized vectors gives cosine similarity
        sim_matrix[i, :] = torch.matmul(X_norm[i].unsqueeze(0), X_norm.t()).squeeze(0)
    
    # Set diagonal to 0 (document similarity to itself is not needed)
    sim_matrix.fill_diagonal_(-1)
    return sim_matrix


def compute_pairwise_cosine_distance_torch(X, device):
    """
    Compute pairwise cosine distance between rows of a matrix X using PyTorch.
    cosine_distance(X_i, X_j) = 1 - cosine_similarity(X_i, X_j)
    This version computes the matrix row-by-row to be more memory-efficient.
    Args:
        X (torch.Tensor): Input tensor of shape (n_docs, n_features) where rows are vectors.
        device (torch.device): The device (CPU or CUDA) to perform calculations on.
    Returns:
        torch.Tensor: Pairwise cosine distance matrix of shape (n_docs, n_docs).
    """
    X = X.to(device)
    n_docs, n_features = X.shape
    # Normalize rows to unit length
    X_norm = torch.nn.functional.normalize(X, p=2, dim=1)
    dist_matrix = torch.zeros((n_docs, n_docs), device=device)
    
    print(f"  Calculating cosine distance row-by-row for {n_docs} documents...")
    for i in tqdm(range(n_docs), desc="  Cosine Dist Row", unit="doc", leave=False, dynamic_ncols=True):
        # Get similarity (dot product of normalized vectors)
        sim = torch.matmul(X_norm[i].unsqueeze(0), X_norm.t()).squeeze(0)
        # Convert to distance: 1 - similarity
        dist_matrix[i, :] = 1 - sim
    
    # Set diagonal to 0 (distance to self is 0)
    dist_matrix.fill_diagonal_(0)
    return dist_matrix


def compute_precision_at_k(retrieved_indices, query_labels, all_labels, k_values=[1, 5, 10]):
    """Compute precision@k for the retrieved documents."""
    results = {}
    query_label = query_labels
    
    for k in k_values:
        if k > len(retrieved_indices):
            continue
        top_k_indices = retrieved_indices[:k]
        top_k_labels = all_labels[top_k_indices]
        precision = np.mean(top_k_labels == query_label)
        results[f'precision@{k}'] = precision
        
    return results

def apply_subsetting(labels, retrieval_representation, subset_size):
    # Apply subsetting for a even number of documents per label
    label_to_indices = defaultdict(list)
    for i, label in enumerate(labels):
        label_to_indices[label].append(i)
        
    min_count = min(len(indices) for indices in label_to_indices.values())
    if subset_size < len(label_to_indices) * min_count:
        docs_per_label = subset_size // len(label_to_indices)
        min_count = min(min_count, docs_per_label)
        
    subset_indices = sum([indices[:min_count] for indices in label_to_indices.values()], [])
    labels = labels[subset_indices]
    retrieval_representation = retrieval_representation[subset_indices]
    return labels, retrieval_representation


def main(args):
    labels = np.loadtxt(args.label_file)
    similarity_matrix: np.ndarray | None = None # Initialize similarity matrix
    
    # Determine device (CUDA or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    ignore_indices = []
    with open(os.path.join(args.bow_corpus_path), 'r', encoding='utf-8') as f:
        bow_corpus = [line.strip().split() for line in f]
        ignore_indices = [i for i, doc in enumerate(bow_corpus) if len(doc) == 0 or doc == ['null']]
        labels = np.array([int(label) for i, label in enumerate(labels) if i not in ignore_indices])
        bow_corpus = [doc for i, doc in enumerate(bow_corpus) if i not in ignore_indices]

    if args.method == 'topic_distribution':
        output = torch.load(args.data_file, weights_only=False)
        if 'topic-document-matrix' not in output.keys():
            raise ValueError(f"topic-document-matrix not found in output keys at {args.data_file}")
        topic_distribution_np: np.ndarray = output['topic-document-matrix'].transpose()
        
        # Ensure topic_distribution_np aligns with filtered labels
        if topic_distribution_np.shape[0] != len(labels):
            # This initial check assumes topic_distribution_np might be larger due to not having ignore_indices applied yet.
            original_td_shape_0 = topic_distribution_np.shape[0]
            original_labels_len = len(labels) # This is already filtered by ignore_indices if subsetting is not applied yet

            # Determine the expected number of documents after ignore_indices filtering (before subsetting)
            # We need to know the original number of documents before ignore_indices were applied to labels.
            # This is tricky because labels are already filtered. We assume topic_distribution_np is from the unfiltered corpus.
            # Let's reload the original labels count to correctly apply ignore_indices to topic_distribution_np
            original_labels_for_filtering = np.loadtxt(args.label_file)
            temp_ignore_indices = [i for i, doc_text in enumerate(open(os.path.join(args.bow_corpus_path), 'r', encoding='utf-8')) if len(doc_text.strip().split()) == 0 or doc_text.strip().split() == ['null']]

            if topic_distribution_np.shape[0] > (len(original_labels_for_filtering) - len(temp_ignore_indices)):
                 # This implies topic_distribution_np is larger than the count of valid documents from bow_corpus
                 # Apply ignore_indices to topic_distribution_np before subsetting
                n_original_docs_from_td = topic_distribution_np.shape[0]
                # Re-calculate ignore_indices based on the original corpus that topic_distribution_np corresponds to.
                # We assume that the original bow_corpus (unfiltered) had n_original_docs_from_td documents.
                # This part is complex because we don't have the original unfiltered bow_corpus easily accessible here to derive ignore_indices for topic_distribution_np.
                # Let's assume ignore_indices are applicable if lengths match the unfiltered source of topic_distribution_np
                # For simplicity, we will assume the pre-loaded ignore_indices can be used if max(ignore_indices) is valid
                if max(temp_ignore_indices) < n_original_docs_from_td:
                    valid_rows_mask = np.ones(n_original_docs_from_td, dtype=bool)
                    valid_rows_mask[temp_ignore_indices] = False
                    topic_distribution_np = topic_distribution_np[valid_rows_mask, :]
                else:
                     raise ValueError(\
                        f"Max index in temp_ignore_indices ({max(temp_ignore_indices)}) is out of bounds for topic_distribution_np " \
                        f"which has {n_original_docs_from_td} documents. Cannot apply ignore_indices directly."\
                    )

            # Final check after all filtering and potential subsetting
            if topic_distribution_np.shape[0] != len(labels):
                raise ValueError(\
                    f"Shape mismatch between topic distributions ({topic_distribution_np.shape[0]}) "\
                    f"and labels ({len(labels)}) after ignore_indices and subsetting. "\
                    f"Original TD shape: {original_td_shape_0}, original labels (after ignore): {original_labels_len}, subset_size: {args.subset_size}"\
                )
        
        if args.subset_size > 0:
            labels, topic_distribution_np = apply_subsetting(labels, topic_distribution_np, args.subset_size)

        assert topic_distribution_np.shape[0] == len(labels), \
            f"topic-document-matrix shape {topic_distribution_np.shape[0]} does not match number of labels {len(labels)} after subsetting"
        
        print("Computing pairwise KL divergence using PyTorch...")
        # Convert to tensor and compute KL divergence
        topic_distribution_tensor = torch.from_numpy(topic_distribution_np).float()
        kl_matrix_torch = compute_pairwise_kl_divergence_torch(topic_distribution_tensor, device)
        similarity_matrix = kl_matrix_torch.cpu().numpy() # Move back to CPU and convert to NumPy
        print("KL divergence computation complete.")

    elif args.method == 'next_word_probs':
        probabilities_files = [f for f in os.listdir(args.probabilities_path) if f.endswith('.pt')]
        probabilities_files.sort(key=lambda x: int(x.split('.')[0]))
        probabilities = [torch.tensor(torch.load(os.path.join(args.probabilities_path, f), weights_only=False)) for f in probabilities_files]
        probabilities = torch.vstack(probabilities)
        probabilities = torch.softmax(probabilities, dim=-1)
        valid_rows_mask = np.ones(len(probabilities), dtype=bool)
        valid_rows_mask[ignore_indices] = False
        probabilities = probabilities[valid_rows_mask]
        if probabilities.shape[0] != len(labels):
            raise ValueError(f"Shape mismatch between probabilities ({probabilities.shape[0]}) and labels ({len(labels)}).")
        similarity_matrix = compute_pairwise_kl_divergence_torch(probabilities, device).cpu().numpy()

    elif args.method == 'hidden_states':
        hidden_states_files = [f for f in os.listdir(args.hidden_state_path) if f.endswith('.pt')]
        hidden_states_files.sort(key=lambda x: int(x.split('.')[0]))
        hidden_states = [torch.tensor(torch.load(os.path.join(args.hidden_state_path, f), weights_only=False)) for f in hidden_states_files]
        hidden_states = torch.vstack(hidden_states)
        valid_rows_mask = np.ones(len(hidden_states), dtype=bool)
        valid_rows_mask[ignore_indices] = False
        hidden_states = hidden_states[valid_rows_mask]
        if hidden_states.shape[0] != len(labels):
            raise ValueError(f"Shape mismatch between probabilities ({hidden_states.shape[0]}) and labels ({len(labels)}).")
        similarity_matrix = compute_pairwise_cosine_distance_torch(hidden_states, device).cpu().numpy()
    elif args.method == 'bow':
        import pdb; pdb.set_trace()
        bow_corpus_tensor = torch.from_numpy(np.array(bow_corpus)).float()
        similarity_matrix = compute_pairwise_cosine_distance_torch(bow_corpus_tensor, device).cpu().numpy()
    else:
        raise NotImplementedError(f"Method {args.method} not implemented")

    # Check if similarity matrix was computed
    if similarity_matrix is None:
        raise ValueError(f"Similarity matrix was not computed. Check implementation for method {args.method}.")

    # Check if it contains NaNs (indicating a placeholder was hit without full implementation)
    if np.isnan(similarity_matrix).all():
         raise NotImplementedError(f"Similarity calculation for method {args.method} resulted in all NaNs. Please implement it fully.")

    n_docs = similarity_matrix.shape[0]
    k_values = [1, 5, 10]
    max_k = max(k_values)
    
    precision_results = {k: [] for k in k_values}
    
    for i in tqdm(range(n_docs), desc="Calculating precision for each document"):
        # Get indices of documents sorted by similarity_matrix values (ascending order, smaller is better)
        retrieved_indices = np.argsort(similarity_matrix[i])
        # Remove self (which should have 0 distance/divergence and be first)
        retrieved_indices = retrieved_indices[retrieved_indices != i][:max_k]
        
        precisions = compute_precision_at_k(retrieved_indices, labels[i], labels, k_values)
        
        for k_val, precision in precisions.items(): # Renamed k to k_val to avoid conflict
            precision_results[int(k_val.split('@')[1])].append(precision)
    
    for k in k_values:
        if precision_results[k]:
            avg_precision = np.mean(precision_results[k])
            print(f"Average Precision@{k}: {avg_precision:.4f}")
        else:
            print(f"No results to average for Precision@{k} (perhaps not enough documents or k is too large).")
    return precision_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bow_corpus_path", type=str, default="data/20_newsgroups_Llama-3.2-1B-Instruct_vocab_2000_last/bow_dataset.txt")
    parser.add_argument("--data_path", type=str, default="results/20_newsgroups/combined_K100/")
    parser.add_argument("--probabilities_path", type=str, default="data/20_newsgroups_Llama-3.2-3B-Instruct_vocab_2000_last/processed_dataset/next_word_probs")
    parser.add_argument("--hidden_state_path", type=str, default="data/20_newsgroups_Llama-3.2-3B-Instruct_vocab_2000_last/processed_dataset/input_embeddings/28")
    parser.add_argument("--label_file", type=str, default="data/20_newsgroups_Llama-3.2-3B-Instruct_vocab_2000_last/numeric_labels.txt")
    parser.add_argument("--method", type=str, default='topic_distribution', choices=['topic_distribution', 'next_word_probs', 'hidden_states', 'bow'])
    parser.add_argument("--subset_size", type=int, default=-1, help="Number of documents to use for a subset. Default is -1 (use all documents).")
    args = parser.parse_args()
    datasets = ['20_newsgroups', 'stackoverflow', 'dbpedia_14']
    label_files = ['data/20_newsgroups_Llama-3.2-1B-Instruct_vocab_2000_last/numeric_labels.txt',
                   'data/stackoverflow_Llama-3.2-1B-Instruct_vocab_2000_last/numeric_labels.txt',
                   'data/dbpedia_14_Llama-3.2-1B-Instruct_vocab_2000_last/numeric_labels.txt']
    bow_corpus_paths = ['data/20_newsgroups_Llama-3.2-1B-Instruct_vocab_2000_last/bow_dataset.txt',
                        'data/stackoverflow_Llama-3.2-1B-Instruct_vocab_2000_last/bow_dataset.txt',
                        'data/dbpedia_14_Llama-3.2-1B-Instruct_vocab_2000_last/bow_dataset.txt']
    # baselines = ['lda', 'prodlda', 'combined', 'zeroshot', 'etm', 'bertopic']
    baselines = ['fastopic']
    # datasets = ['stackoverflow']
    # label_files = ['data/stackoverflow_Llama-3.2-1B-Instruct_vocab_2000_last/numeric_labels.txt']
    # bow_corpus_paths = ['data/stackoverflow_Llama-3.2-1B-Instruct_vocab_2000_last/bow_dataset.txt']
    # ours = ['Llama-3.2-11B-Vision-Instruct-CE', 'Llama-3.2-11B-Vision-Instruct-KL']
    topic_models = baselines

    for dataset, label_file, bow_corpus_path in list(zip(datasets, label_files, bow_corpus_paths))[2:]:
        print(f"Computing retrieval results for {dataset} dataset\n\n\n")
        if dataset == 'dbpedia_14':
            args.subset_size = 70000
        args.bow_corpus_path = bow_corpus_path
        args.label_file = label_file
        for topic_model in topic_models:
            table_cell = []
            for num_topics in [25, 50, 75, 100]:
                if topic_model.startswith('Llama'):
                    llm_name = '-'.join(topic_model.split('-')[:-1])
                    loss_method = topic_model.split('-')[-1]
                    args.data_path = f"results/{dataset}/{llm_name}/{num_topics}_{loss_method}"
                else:
                    args.data_path = f"results/{dataset}/{topic_model}_K{num_topics}/"
                all_seed_results = []
                seed_dirs = [d for d in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, d))]
                # Store results from all seeds
                all_results = defaultdict(list)
                for seed_dir in seed_dirs:
                    print(f"\nProcessing seed directory: {seed_dir}")
                    args.data_file = os.path.join(args.data_path, seed_dir, 'model_output.pt')
                    results = main(args)
                    for k, precisions in results.items():
                        all_results[k].append(np.mean(precisions))

                # Calculate and print average across all seeds
                if all_results:
                    print("\n" + "="*50)
                    print(f"AVERAGE RESULTS ACROSS {len(seed_dirs)} SEEDS for {topic_model} with {num_topics} topics")
                    print(f"Data Path: {args.data_path}")
                    print("="*50)
                    
                    for k in sorted(all_results.keys()):
                        avg_precision = np.mean(all_results[k])
                        std_precision = np.std(all_results[k])
                        table_cell.append(avg_precision)
                        print(f"Average Precision@{k}: {avg_precision:.4f} ± {std_precision:.4f}")
                else:
                    print("No results to average (perhaps not enough documents or k is too large).")
            print(f"Table row for {topic_model} with topics: [25, 50, 75, 100] on {dataset} dataset")
            print('\t'.join(map(str, table_cell)))