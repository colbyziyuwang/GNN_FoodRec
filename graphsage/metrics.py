# Implement metrics

import numpy as np

def precision_at_k(recommended, relevant, k):
    """
    Compute precision at k.
    Parameters:
    - recommended: List of recommended items.
    - relevant: Set of ground truth relevant items.
    - k: Number of top recommendations to consider.

    Returns:
    - Precision at k.
    """
    recommended_at_k = recommended[:k]
    relevant_at_k = set(recommended_at_k) & set(relevant)
    return len(relevant_at_k) / k

def recall_at_k(recommended, relevant, k):
    """
    Compute recall at k.
    Parameters:
    - recommended: List of recommended items.
    - relevant: Set of ground truth relevant items.
    - k: Number of top recommendations to consider.

    Returns:
    - Recall at k.
    """
    recommended_at_k = recommended[:k]
    relevant_at_k = set(recommended_at_k) & set(relevant)
    return len(relevant_at_k) / len(relevant) if len(relevant) > 0 else 0

def ndcg_at_k(recommended, relevant, k):
    """
    Compute normalized discounted cumulative gain (NDCG) at k.
    Parameters:
    - recommended: List of recommended items.
    - relevant: Set of ground truth relevant items.
    - k: Number of top recommendations to consider.

    Returns:
    - NDCG at k.
    """
    recommended_at_k = recommended[:k]
    dcg = sum([1 / np.log2(i + 2) if recommended_at_k[i] in relevant else 0 for i in range(k)])
    idcg = sum([1 / np.log2(i + 2) for i in range(min(len(relevant), k))])
    return dcg / idcg if idcg > 0 else 0

def hit_rate_at_k(recommended, relevant, k):
    """
    Compute hit rate at k.
    Parameters:
    - recommended: List of recommended items.
    - relevant: Set of ground truth relevant items.
    - k: Number of top recommendations to consider.

    Returns:
    - 1 if at least one relevant item is in the top-k recommendations, 0 otherwise.
    """
    recommended_at_k = recommended[:k]
    return 1 if set(recommended_at_k) & set(relevant) else 0
