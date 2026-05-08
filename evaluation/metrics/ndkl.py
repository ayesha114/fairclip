"""
=============================================================================
NDKL Metric — Normalized Discounted KL Divergence
=============================================================================
NDKL measures fairness in ranked retrieval results.
Used by Zhang et al. CVPR 2025.
Lower NDKL = fairer model.

Gives more weight to positions near the top of the ranking
(discounting by position), which reflects that users pay more
attention to top results.
=============================================================================
"""

import numpy as np
import torch


def compute_ndkl(
    similarity_matrix: torch.Tensor,
    group_labels: torch.Tensor,
    k: int = 10,
) -> float:
    """
    Compute NDKL fairness metric.

    Args:
        similarity_matrix: [N_queries, N_items] cosine similarity scores
        group_labels: [N_items] demographic group labels
        k: ranking depth to consider

    Returns:
        ndkl: float (lower = fairer)
    """
    n_queries = similarity_matrix.shape[0]
    unique_groups = group_labels.unique()
    n_groups = len(unique_groups)

    if n_groups < 2:
        return 0.0

    # Ideal uniform distribution over groups
    p_ideal = {g.item(): 1.0 / n_groups for g in unique_groups}

    # Discounting weights (position-based)
    # Weight at position i = 1/log2(i+2) (standard DCG-style discounting)
    discount = np.array([1.0 / np.log2(i + 2) for i in range(k)])
    Z = discount.sum()  # normalization factor

    ndkl_scores = []

    for q in range(n_queries):
        scores = similarity_matrix[q]
        n_items = len(scores)
        actual_k = min(k, n_items)
        topk_indices = scores.topk(min(actual_k, len(scores))).indices
        topk_labels = group_labels[topk_indices]

        # Compute KL divergence at each position
        kl_sum = 0.0
        cumulative_counts = {g.item(): 0 for g in unique_groups}

        for pos in range(actual_k):
            label = topk_labels[pos].item()
            cumulative_counts[label] += 1

            # Empirical distribution up to this position
            total = pos + 1
            kl = 0.0
            for g in unique_groups:
                g = g.item()
                p_empirical = cumulative_counts[g] / total
                p_id = p_ideal[g]
                if p_empirical > 0:
                    kl += p_empirical * np.log(p_empirical / p_id)

            kl_sum += discount[pos] * kl

        ndkl_scores.append(kl_sum / Z)

    return float(np.mean(ndkl_scores))
