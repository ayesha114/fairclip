"""
ABLE Metric — Alignment and Bias Level Evaluation
Zhang et al. CVPR 2025 combined metric.
Higher ABLE = better overall model.
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

    image_embeddings: [N, D] — one per image
    text_embeddings:  [P, D] — one per demographic prompt (P = n_groups)
    group_labels:     [N]    — demographic label for each image
    """
    n_images = len(image_embeddings)
    n_prompts = len(text_embeddings)

    # Similarity matrix: [N_images, N_prompts]
    sim = image_embeddings.float() @ text_embeddings.float().T

    # VL Alignment = how often the model picks the correct demographic prompt
    # For each image, correct prompt = index matching its group label
    correct = 0
    for i in range(n_images):
        label = group_labels[i].item()
        if label < 0 or label >= n_prompts:
            continue
        top1 = sim[i].argmax().item()
        if top1 == label:
            correct += 1

    valid = (group_labels >= 0) & (group_labels < n_prompts)
    n_valid = valid.sum().item()
    vl_alignment = correct / n_valid if n_valid > 0 else 0.0

    # MaxSkew: use image→prompt similarities, groups = image group labels
    valid_sim = sim[valid]
    valid_labels = group_labels[valid]
    maxskew = compute_maxskew(valid_sim, valid_labels, k=min(k, n_prompts))

    # Normalize MaxSkew to [0, 1]
    max_possible = np.log(max(k, 2))
    bias_level = min(maxskew / max_possible, 1.0)

    able = vl_alignment * (1.0 - bias_level)

    return {
        "able": able,
        "vl_alignment": vl_alignment,
        "bias_level": bias_level,
        "maxskew": maxskew,
    }
