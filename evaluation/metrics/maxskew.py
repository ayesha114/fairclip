"""
=============================================================================
MaxSkew Metric
=============================================================================
MaxSkew measures the maximum skew between demographic groups in retrieval.

Used by Zhang et al. CVPR 2025 as a primary fairness metric.
Lower MaxSkew = fairer model.

Formula:
    For each query, compute the ratio of retrieved items from each group.
    MaxSkew@K = max over all queries of |log(p_group / p_uniform)|
    where p_uniform = 1/K (ideal uniform distribution)
=============================================================================
"""

import numpy as np
import torch


def compute_maxskew(
    similarity_matrix: torch.Tensor,
    group_labels: torch.Tensor,
    k: int = 5,
) -> float:
    """
    Compute MaxSkew@K fairness metric.

    Args:
        similarity_matrix: [N_queries, N_items] cosine similarity scores
        group_labels: [N_items] demographic group label for each item
        k: top-K to consider

    Returns:
        maxskew: float (lower = fairer, 0 = perfectly fair)
    """
    n_queries = similarity_matrix.shape[0]
    unique_groups = group_labels.unique()
    n_groups = len(unique_groups)

    if n_groups < 2:
        return 0.0

    # Ideal proportion for each group (uniform = 1/n_groups)
    p_uniform = 1.0 / n_groups

    skews = []
    for q in range(n_queries):
        # Get top-K items for this query
        scores = similarity_matrix[q]
        actual_k = min(k, len(scores))
        topk_indices = scores.topk(actual_k).indices

        # Count how many items from each group appear in top-K
        topk_labels = group_labels[topk_indices]

        for group in unique_groups:
            group_count = (topk_labels == group).sum().item()
            p_group = group_count / k

            # Avoid log(0)
            if p_group == 0:
                p_group = 1e-10

            # Skew = |log(actual / expected)|
            skew = abs(np.log(p_group / p_uniform))
            skews.append(skew)

    return float(np.max(skews)) if skews else 0.0
