"""
=============================================================================
FairCLIP — Complete Evaluation Metrics
=============================================================================
ALL metrics from your proposal document (Section 12).

Performance Metrics (Section 12.1):
    - Accuracy
    - F1-Score (Precision + Recall)
    - Convergence Speed

Fairness Metrics (Section 12.2):
    - Demographic Parity Gap (DPG)
    - Equalized Odds Difference (EOD)
    - Facet Bias Score

Efficiency Metrics (Section 12.3):
    - Computational Cost
    - Training Time per Epoch

Plus standard CLIP debiasing metrics:
    - MaxSkew@K
    - NDKL
    - ABLE
    - TR@K, IR@K (Retrieval)

These are ALL the numbers that go in your paper tables.
=============================================================================
"""

import time
import logging
import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


# =============================================================================
# Section 12.1 — Performance Metrics
# =============================================================================

def compute_accuracy(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
) -> float:
    """
    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    Measures how often the model correctly aligns image-text pairs.

    Args:
        predictions: [N] binary predictions (0 or 1)
        ground_truth: [N] binary ground truth labels (0 or 1)

    Returns:
        accuracy: float between 0 and 1
    """
    if len(predictions) == 0:
        return 0.0

    correct = (predictions == ground_truth).sum().item()
    total = len(predictions)
    accuracy = correct / total

    return float(accuracy)


def compute_precision_recall_f1(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
) -> dict:
    """
    Compute Precision, Recall, and F1-Score.

    F1-Score is suitable for imbalanced demographic data.

    Formulas from your proposal:
        Precision = TP / (TP + FP)
        Recall    = TP / (TP + FN)
        F1        = 2 × Precision × Recall / (Precision + Recall)

    Args:
        predictions: [N] binary predictions
        ground_truth: [N] binary ground truth

    Returns:
        dict with precision, recall, f1
    """
    predictions = predictions.long()
    ground_truth = ground_truth.long()

    # True Positives, False Positives, False Negatives
    tp = ((predictions == 1) & (ground_truth == 1)).sum().item()
    fp = ((predictions == 1) & (ground_truth == 0)).sum().item()
    fn = ((predictions == 0) & (ground_truth == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
    }


def compute_convergence_speed(
    accuracy_history: list[float],
    target_accuracy: float = 0.7,
) -> int:
    """
    Convergence Speed = minimum epochs to reach target accuracy.

    Formula from your proposal:
        R_conv = min{r | A_r >= A_target}

    Args:
        accuracy_history: list of accuracy values per epoch
        target_accuracy: the threshold accuracy to reach

    Returns:
        convergence_epoch: int (epoch number when target was reached)
                          Returns -1 if target was never reached.
    """
    for epoch, acc in enumerate(accuracy_history, start=1):
        if acc >= target_accuracy:
            return epoch
    return -1  # Did not converge to target


# =============================================================================
# Section 12.2 — Fairness Metrics
# =============================================================================

def compute_dpg_full(
    predictions: torch.Tensor,
    attribute_labels: torch.Tensor,
) -> float:
    """
    Demographic Parity Gap (DPG).

    Formula from your proposal:
        DPG = |P(Y_hat=1 | A=a1) - P(Y_hat=1 | A=a2)|

    Promotes equity among demographic groups.
    Lower DPG = fairer model.

    For multiple groups (more than 2), we compute the maximum pairwise gap.

    Args:
        predictions: [N] binary predictions (0 or 1)
        attribute_labels: [N] demographic group labels

    Returns:
        dpg: float (lower = fairer, 0 = perfect parity)
    """
    unique_groups = attribute_labels.unique()
    valid_groups = [g for g in unique_groups if g.item() != -1]

    if len(valid_groups) < 2:
        return 0.0

    # Compute P(Y_hat=1) for each group
    positive_rates = {}
    for group in valid_groups:
        mask = attribute_labels == group
        if mask.sum() == 0:
            continue
        group_preds = predictions[mask].float()
        positive_rates[group.item()] = group_preds.mean().item()

    if len(positive_rates) < 2:
        return 0.0

    # Maximum pairwise gap across all group pairs
    rates = list(positive_rates.values())
    max_gap = 0.0
    for i in range(len(rates)):
        for j in range(i + 1, len(rates)):
            gap = abs(rates[i] - rates[j])
            max_gap = max(max_gap, gap)

    return float(max_gap)


def compute_eod_full(
    predictions: torch.Tensor,
    true_labels: torch.Tensor,
    attribute_labels: torch.Tensor,
) -> float:
    """
    Equalized Odds Difference (EOD).

    Formula from your proposal:
        EOD = |TPR_a1 - TPR_a2| + |FPR_a1 - FPR_a2|

    Fairness based on comparing true and false positive rates.
    Lower EOD = fairer model.

    Args:
        predictions: [N] binary predictions
        true_labels: [N] ground truth labels
        attribute_labels: [N] demographic group labels

    Returns:
        eod: float (lower = fairer)
    """
    unique_groups = attribute_labels.unique()
    valid_groups = [g for g in unique_groups if g.item() != -1]

    if len(valid_groups) < 2:
        return 0.0

    tpr_per_group = {}
    fpr_per_group = {}

    for group in valid_groups:
        mask = attribute_labels == group
        if mask.sum() == 0:
            continue

        g_preds = predictions[mask].float()
        g_true = true_labels[mask].float()

        # True Positive Rate = TP / (TP + FN)
        pos_mask = g_true == 1
        tpr = (
            (g_preds[pos_mask] == 1).float().mean().item()
            if pos_mask.sum() > 0 else 0.0
        )

        # False Positive Rate = FP / (FP + TN)
        neg_mask = g_true == 0
        fpr = (
            (g_preds[neg_mask] == 1).float().mean().item()
            if neg_mask.sum() > 0 else 0.0
        )

        tpr_per_group[group.item()] = tpr
        fpr_per_group[group.item()] = fpr

    if len(tpr_per_group) < 2:
        return 0.0

    # Maximum pairwise EOD across all group pairs
    groups_list = list(tpr_per_group.keys())
    max_eod = 0.0

    for i in range(len(groups_list)):
        for j in range(i + 1, len(groups_list)):
            g1, g2 = groups_list[i], groups_list[j]
            eod = (
                abs(tpr_per_group[g1] - tpr_per_group[g2])
                + abs(fpr_per_group[g1] - fpr_per_group[g2])
            )
            max_eod = max(max_eod, eod)

    return float(max_eod)


def compute_facet_bias_score(
    bias_values_per_facet: dict,
) -> float:
    """
    Facet Bias Score.

    Formula from your proposal:
        FacetBiasScore = (1/N) * sum_i |Bias_i|

    Measures fairness across demographic facets
    (gender, race, age simultaneously).

    Args:
        bias_values_per_facet: dict mapping facet name -> bias value
                               e.g., {"gender": 0.15, "race": 0.22, "age": 0.09}
                               Each value is the DPG or MaxSkew for that facet.

    Returns:
        facet_bias_score: float (lower = fairer overall)
    """
    if not bias_values_per_facet:
        return 0.0

    bias_values = [abs(v) for v in bias_values_per_facet.values()]
    return float(sum(bias_values) / len(bias_values))


# =============================================================================
# Section 12.3 — Efficiency Metrics
# =============================================================================

class EfficiencyTracker:
    """
    Tracks computational cost and energy consumption during training.

    Formula from your proposal:
        C_comp = E_i × f_i × T_i

    Where:
        E_i = number of epochs
        f_i = computational cost per epoch (FLOPs approximation)
        T_i = execution time per epoch (seconds)
    """

    def __init__(self):
        self.epoch_times = []          # seconds per epoch
        self.total_start_time = None
        self.epoch_start_time = None
        self.n_epochs = 0
        self.flops_per_epoch = None    # filled in after first batch

    def start_training(self):
        """Call at the start of training."""
        self.total_start_time = time.time()

    def start_epoch(self):
        """Call at the start of each epoch."""
        self.epoch_start_time = time.time()

    def end_epoch(self):
        """Call at the end of each epoch."""
        if self.epoch_start_time is None:
            return
        elapsed = time.time() - self.epoch_start_time
        self.epoch_times.append(elapsed)
        self.n_epochs += 1
        log.info(f"Epoch {self.n_epochs} time: {elapsed:.1f}s")

    def estimate_flops_per_batch(self, model, batch_size: int) -> int:
        """
        Approximate FLOPs for one forward pass.
        Uses a rough estimate based on embedding dimension and batch size.
        """
        d = model.get_embedding_dim()
        # Rough FLOPs: 2 × batch × seq_len × d^2 for transformer attention
        # This is a simplified estimate; exact count requires profiling
        flops = 2 * batch_size * 197 * (d ** 2)  # 197 = ViT-B/32 sequence length
        self.flops_per_epoch = flops
        return flops

    def compute_cost(self) -> dict:
        """
        Compute all efficiency metrics.

        Returns:
            dict with:
                total_time: total training time in seconds
                avg_epoch_time: average time per epoch
                n_epochs: number of completed epochs
                computational_cost: E × f × T estimate
        """
        if not self.epoch_times:
            return {
                "total_time_s": 0.0,
                "avg_epoch_time_s": 0.0,
                "n_epochs": 0,
                "computational_cost": 0.0,
            }

        total_time = sum(self.epoch_times)
        avg_time = total_time / len(self.epoch_times)

        # C_comp = E_i × f_i × T_i
        # If FLOPs not measured, use total time as proxy
        f_i = self.flops_per_epoch if self.flops_per_epoch else 1.0
        computational_cost = self.n_epochs * f_i * avg_time

        return {
            "total_time_s": round(total_time, 2),
            "avg_epoch_time_s": round(avg_time, 2),
            "n_epochs": self.n_epochs,
            "computational_cost": computational_cost,
        }


# =============================================================================
# Master function: compute ALL metrics at once
# =============================================================================

def compute_all_metrics(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    demographic_labels: torch.Tensor,
    attribute: str,
    accuracy_history: list = None,
    efficiency_tracker: EfficiencyTracker = None,
) -> dict:
    """
    Compute every metric from your proposal in one function call.

    This is what gets called after each training run to produce
    the full results table for your paper.

    Args:
        image_embeddings: [N, D] L2-normalized
        text_embeddings: [N, D] L2-normalized (demographic prompts)
        demographic_labels: [N] group labels
        attribute: "gender", "age", or "race"
        accuracy_history: list of epoch accuracies (for convergence speed)
        efficiency_tracker: EfficiencyTracker instance (for timing)

    Returns:
        dict with ALL metrics — paste directly into paper table
    """
    from evaluation.metrics.maxskew import compute_maxskew
    from evaluation.metrics.ndkl import compute_ndkl
    from evaluation.metrics.able import compute_able
    from evaluation.retrieval import compute_recall_at_k

    results = {"attribute": attribute}

    # Filter valid labels
    valid_mask = demographic_labels != -1
    valid_img = image_embeddings[valid_mask]
    valid_labels = demographic_labels[valid_mask]

    # Similarity matrix
    sim = valid_img.float() @ text_embeddings.float().T

    # --- Retrieval-based predictions for classification metrics ---
    top1_pred = sim.argmax(dim=1)  # predicted group for each image

    # Binary: correct if predicted group matches true group
    preds_binary = (top1_pred == valid_labels).long()
    true_binary = torch.ones(len(valid_labels), dtype=torch.long)

    # ─── Performance Metrics (Section 12.1) ─────────────────────────────────
    results["accuracy"] = round(
        compute_accuracy(preds_binary, true_binary), 4
    )

    prf = compute_precision_recall_f1(preds_binary, true_binary)
    results.update(prf)  # adds precision, recall, f1

    if accuracy_history:
        results["convergence_epoch"] = compute_convergence_speed(
            accuracy_history, target_accuracy=0.7
        )

    # ─── Fairness Metrics (Section 12.2) ─────────────────────────────────────
    results["dpg"] = round(
        compute_dpg_full(preds_binary, valid_labels), 4
    )
    results["eod"] = round(
        compute_eod_full(preds_binary, true_binary, valid_labels), 4
    )

    # ─── Standard CLIP debiasing metrics ─────────────────────────────────────
    results["maxskew_5"] = round(
        compute_maxskew(sim, valid_labels, k=5), 4
    )
    results["ndkl"] = round(
        compute_ndkl(sim, valid_labels, k=10), 4
    )

    able_result = compute_able(valid_img, text_embeddings.float(), valid_labels, k=5)
    results["able"] = round(able_result["able"], 4)
    results["vl_alignment"] = round(able_result["vl_alignment"], 4)
    results["bias_level"] = round(able_result["bias_level"], 4)

    # ─── Retrieval metrics (RQ4) ─────────────────────────────────────────────
    n_eval = min(500, len(valid_img), len(text_embeddings))
    recall = compute_recall_at_k(
        valid_img[:n_eval],
        text_embeddings[:n_eval] if len(text_embeddings) >= n_eval else text_embeddings,
        k_values=[1, 5, 10],
    )
    results.update(recall)

    # ─── Efficiency metrics (Section 12.3) ───────────────────────────────────
    if efficiency_tracker:
        cost = efficiency_tracker.compute_cost()
        results.update(cost)

    return results
