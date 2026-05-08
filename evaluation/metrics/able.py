"""
=============================================================================
ABLE Metric — Alignment and Bias Level Evaluation
=============================================================================
ABLE is the combined metric proposed by Zhang et al. CVPR 2025.
It balances BOTH fairness AND accuracy in one score.
Higher ABLE = better overall model.

Formula:
    ABLE = VL_Alignment × (1 - Bias_Level)

Where:
    VL_Alignment = text-to-image retrieval accuracy (R@1)
    Bias_Level   = normalized MaxSkew (between 0 and 1)

ABLE is the PRIMARY metric for comparing FairCLIP vs Zhang.
If your ABLE > Zhang's ABLE, your method is better.
=============================================================================
"""

import numpy as np
import torch
from evaluation.metrics.maxskew import compute_maxskew


def compute_able(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    group_labels: torch.Tensor,
    k: int = 5,
) -> dict:
    """
    Compute ABLE metric.

    Args:
        image_embeddings: [N, D] L2-normalized image embeddings
        text_embeddings: [N, D] L2-normalized text embeddings
        group_labels: [N] demographic group labels for images
        k: top-K for retrieval and MaxSkew

    Returns:
        dict with keys:
            able: float (main metric, higher = better)
            vl_alignment: float (retrieval accuracy)
            bias_level: float (normalized bias, lower = better)
            maxskew: float (raw MaxSkew value)
    """
    n = len(image_embeddings)

    # Compute similarity matrix: [N, N]
    sim = image_embeddings.float() @ text_embeddings.float().T

    # VL Alignment = Text-to-Image Recall@1
    # For each text query, check if the correct image is rank 1
    correct = 0
    for i in range(n):
        text_query_sims = sim[:, i]  # similarities of all images to text i
        top1_idx = text_query_sims.argmax().item()
        if top1_idx == i:
            correct += 1
    vl_alignment = correct / n

    # Compute MaxSkew (image retrieval fairness)
    # Use text as queries, images as items
    maxskew = compute_maxskew(sim.T, group_labels, k=k)

    # Normalize MaxSkew to [0, 1]
    # MaxSkew can theoretically go up to log(K), so we clip and normalize
    max_possible_skew = np.log(k) if k > 1 else 1.0
    bias_level = min(maxskew / max_possible_skew, 1.0)

    # ABLE = alignment × (1 - bias)
    able = vl_alignment * (1.0 - bias_level)

    return {
        "able": able,
        "vl_alignment": vl_alignment,
        "bias_level": bias_level,
        "maxskew": maxskew,
    }
