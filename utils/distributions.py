import numpy as np

def combine_distributions(distributions, weights=None):
    """
    Combine multiple topic distributions into a single distribution via
    (weighted) summation and normalization.
    
    :param distributions: List of 1D NumPy arrays, each representing a topic distribution.
                          Each array should have the same length (number of topics).
    :param weights: List of non-negative floats, same length as `distributions`, which
                    sum to 1. If None, they are set to 1 / number_of_distributions.
    :return: A 1D NumPy array representing the combined topic distribution.
    """
    
    num_distributions = len(distributions)
    if num_distributions == 0:
        return None
    
    # If weights aren't given, use equal weights
    if weights is None:
        weights = [1.0 / num_distributions] * num_distributions
    else:
        # Ensure weights sum to 1
        total = sum(weights)
        if not np.isclose(total, 1.0):
            weights = [w / total for w in weights]
    
    # Weighted sum of distributions
    combined = np.zeros_like(distributions[0], dtype=float)
    for distr, w in zip(distributions, weights):
        combined += w * np.array(distr)
    
    # Normalize to ensure the sum is 1 (handles floating-point issues)
    combined_sum = combined.sum()
    if combined_sum > 0:
        combined /= combined_sum
    
    return combined