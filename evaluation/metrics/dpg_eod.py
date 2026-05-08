"""
=============================================================================
DPG and EOD Fairness Metrics
=============================================================================
These are standard ML fairness metrics used in top-tier papers.

DPG (Demographic Parity Gap):
    The difference in positive prediction rates between groups.
    Lower DPG = fairer model.

EOD (Equalized Odds Difference):
    The difference in true positive rates between groups.
    Lower EOD = fairer model.

Both are used in your Step IX evaluation.
=============================================================================
"""

import numpy as np
import torch


def compute_dpg(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    attribute_labels: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Compute Demographic Parity Gap.

    DPG measures whether the model's retrieval behavior is independent
    of demographic group membership.

    Args:
        embeddings: [N, D] embeddings
        labels: [N] binary class labels (e.g., retrieved or not)
        attribute_labels: [N] demographic group labels
        threshold: similarity threshold for positive prediction

    Returns:
        dpg: float (lower = fairer, 0 = perfect parity)
    """
    unique_groups = attribute_labels.unique()
    if len(unique_groups) < 2:
        return 0.0

    # Compute positive rate per group
    positive_rates = []
    for group in unique_groups:
        mask = attribute_labels == group
        if mask.sum() == 0:
            continue
        group_labels = labels[mask].float()
        positive_rate = group_labels.mean().item()
        positive_rates.append(positive_rate)

    if len(positive_rates) < 2:
        return 0.0

    # DPG = max difference between any two group positive rates
    dpg = max(positive_rates) - min(positive_rates)
    return float(dpg)


def compute_eod(
    predictions: torch.Tensor,
    true_labels: torch.Tensor,
    attribute_labels: torch.Tensor,
) -> float:
    """
    Compute Equalized Odds Difference.

    EOD measures whether the model has equal true positive rates
    across demographic groups.

    Args:
        predictions: [N] binary predictions (0 or 1)
        true_labels: [N] ground truth labels (0 or 1)
        attribute_labels: [N] demographic group labels

    Returns:
        eod: float (lower = fairer)
    """
    unique_groups = attribute_labels.unique()
    if len(unique_groups) < 2:
        return 0.0

    tpr_per_group = []
    for group in unique_groups:
        mask = attribute_labels == group
        if mask.sum() == 0:
            continue

        group_preds = predictions[mask].float()
        group_true = true_labels[mask].float()

        # True positive rate for this group
        positive_mask = group_true == 1
        if positive_mask.sum() == 0:
            continue

        tpr = (group_preds[positive_mask] == 1).float().mean().item()
        tpr_per_group.append(tpr)

    if len(tpr_per_group) < 2:
        return 0.0

    eod = max(tpr_per_group) - min(tpr_per_group)
    return float(eod)
