"""
=============================================================================
FairCLIP — Complete Evaluation Metrics (All Proposal Metrics)
=============================================================================
Correctly implemented metrics for the full evaluation pipeline.

Performance (Section 12.1): Accuracy, F1, Precision, Recall
Fairness   (Section 12.2): DPG, EOD, Facet Bias Score, MaxSkew, NDKL, ABLE
Retrieval:                  TR@K, IR@K
=============================================================================
"""

import time
import logging
import numpy as np
import torch

log = logging.getLogger(__name__)


# =============================================================================
# Performance Metrics (Section 12.1)
# =============================================================================

def compute_accuracy(predictions, ground_truth):
    """Accuracy = correct / total"""
    if len(predictions) == 0:
        return 0.0
    return float((predictions == ground_truth).sum().item() / len(predictions))


def compute_precision_recall_f1(predictions, ground_truth):
    """Precision, Recall, F1"""
    predictions = predictions.long()
    ground_truth = ground_truth.long()
    tp = ((predictions == 1) & (ground_truth == 1)).sum().item()
    fp = ((predictions == 1) & (ground_truth == 0)).sum().item()
    fn = ((predictions == 0) & (ground_truth == 1)).sum().item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(float(precision), 4),
        "recall":    round(float(recall), 4),
        "f1":        round(float(f1), 4),
    }


def compute_convergence_speed(accuracy_history, target_accuracy=0.7):
    """Minimum epochs to reach target accuracy."""
    for epoch, acc in enumerate(accuracy_history, start=1):
        if acc >= target_accuracy:
            return epoch
    return -1


# =============================================================================
# Fairness Metrics (Section 12.2)
# =============================================================================

def compute_dpg_full(predictions, attribute_labels):
    """
    DPG = max pairwise gap in positive prediction rates between groups.
    Lower = fairer.
    """
    unique_groups = [g for g in attribute_labels.unique() if g.item() != -1]
    if len(unique_groups) < 2:
        return 0.0
    rates = []
    for group in unique_groups:
        mask = attribute_labels == group
        if mask.sum() == 0:
            continue
        rates.append(predictions[mask].float().mean().item())
    if len(rates) < 2:
        return 0.0
    return float(max(rates) - min(rates))


def compute_eod_full(predictions, true_labels, attribute_labels):
    """
    EOD = |TPR_a1 - TPR_a2| + |FPR_a1 - FPR_a2|
    Lower = fairer.
    """
    unique_groups = [g for g in attribute_labels.unique() if g.item() != -1]
    if len(unique_groups) < 2:
        return 0.0
    tprs, fprs = [], []
    for group in unique_groups:
        mask = attribute_labels == group
        if mask.sum() == 0:
            continue
        g_preds = predictions[mask].float()
        g_true  = true_labels[mask].float()
        pos = g_true == 1
        neg = g_true == 0
        tpr = (g_preds[pos] == 1).float().mean().item() if pos.sum() > 0 else 0.0
        fpr = (g_preds[neg] == 1).float().mean().item() if neg.sum() > 0 else 0.0
        tprs.append(tpr)
        fprs.append(fpr)
    if len(tprs) < 2:
        return 0.0
    return float(abs(max(tprs) - min(tprs)) + abs(max(fprs) - min(fprs)))


def compute_facet_bias_score(bias_values_per_facet):
    """
    Facet Bias Score = (1/N) * sum |Bias_i|
    Average bias across all demographic facets.
    """
    if not bias_values_per_facet:
        return 0.0
    return float(sum(abs(v) for v in bias_values_per_facet.values()) / len(bias_values_per_facet))


# =============================================================================
# CLIP-specific fairness metrics
# =============================================================================

def compute_maxskew_correct(image_embeddings, group_labels, k=5):
    """
    Correct MaxSkew computation for FairCLIP.

    For each image query, retrieve top-K most similar images.
    Measure how skewed the demographic distribution is in the retrieved set.

    This is the IMAGE-TO-IMAGE retrieval fairness metric.
    Lower = fairer.
    """
    n = len(image_embeddings)
    unique_groups = [g for g in group_labels.unique() if g.item() != -1]
    n_groups = len(unique_groups)
    if n_groups < 2:
        return 0.0

    p_uniform = 1.0 / n_groups

    # Compute full similarity matrix
    sim = image_embeddings.float() @ image_embeddings.float().T  # [N, N]
    sim.fill_diagonal_(-float('inf'))  # exclude self

    actual_k = min(k, n - 1)
    skews = []

    for q in range(n):
        topk_idx = sim[q].topk(actual_k).indices
        topk_labels = group_labels[topk_idx]
        for group in unique_groups:
            count = (topk_labels == group).sum().item()
            p_group = max(count / actual_k, 1e-10)
            skews.append(abs(np.log(p_group / p_uniform)))

    return float(np.max(skews)) if skews else 0.0


def compute_ndkl_correct(image_embeddings, group_labels, k=10):
    """
    Correct NDKL for image-to-image retrieval.
    Lower = fairer.
    """
    n = len(image_embeddings)
    unique_groups = [g for g in group_labels.unique() if g.item() != -1]
    n_groups = len(unique_groups)
    if n_groups < 2:
        return 0.0

    p_ideal = {g.item(): 1.0 / n_groups for g in unique_groups}
    actual_k = min(k, n - 1)
    discount = np.array([1.0 / np.log2(i + 2) for i in range(actual_k)])
    Z = discount.sum()

    sim = image_embeddings.float() @ image_embeddings.float().T
    sim.fill_diagonal_(-float('inf'))

    scores = []
    for q in range(min(n, 500)):  # sample 500 queries for speed
        topk_idx = sim[q].topk(actual_k).indices
        topk_labels = group_labels[topk_idx]
        cumulative = {g.item(): 0 for g in unique_groups}
        kl_sum = 0.0
        for pos in range(actual_k):
            label = topk_labels[pos].item()
            cumulative[label] += 1
            total = pos + 1
            kl = sum(
                (cumulative[g] / total) * np.log((cumulative[g] / total) / p_ideal[g])
                for g in cumulative if cumulative[g] > 0
            )
            kl_sum += discount[pos] * kl
        scores.append(kl_sum / Z)

    return float(np.mean(scores))


def compute_able_correct(image_embeddings, text_embeddings, group_labels, k=5):
    """
    ABLE = VL_Alignment × (1 - Bias_Level)

    VL_Alignment: how well images match their demographic text prompt
    Bias_Level: normalized MaxSkew from image retrieval

    Higher ABLE = better.
    """
    n_images = len(image_embeddings)
    n_prompts = len(text_embeddings)

    # Image-to-text similarity [N, P]
    sim_i2t = image_embeddings.float() @ text_embeddings.float().T

    # VL Alignment: for each image, does top-1 text match the true demographic?
    correct = 0
    valid = 0
    for i in range(n_images):
        label = group_labels[i].item()
        if label < 0 or label >= n_prompts:
            continue
        top1 = sim_i2t[i].argmax().item()
        if top1 == label:
            correct += 1
        valid += 1

    vl_alignment = correct / valid if valid > 0 else 0.0

    # MaxSkew from image-to-image retrieval
    maxskew = compute_maxskew_correct(image_embeddings, group_labels, k=k)
    max_possible = np.log(max(k, 2))
    bias_level = min(maxskew / max_possible, 1.0)

    able = vl_alignment * (1.0 - bias_level)

    return {
        "able": round(float(able), 4),
        "vl_alignment": round(float(vl_alignment), 4),
        "bias_level": round(float(bias_level), 4),
        "maxskew": round(float(maxskew), 4),
    }


# =============================================================================
# Retrieval metrics TR@K, IR@K
# =============================================================================

def compute_recall_at_k(image_embeddings, text_embeddings, k_values=[1, 5, 10]):
    """
    TR@K: for each text, find matching image in top-K
    IR@K: for each image, find matching text in top-K
    Assumes image[i] pairs with text[i].
    """
    n = min(len(image_embeddings), len(text_embeddings))
    img_embs = image_embeddings[:n].float()
    txt_embs = text_embeddings[:n].float()
    sim = img_embs @ txt_embs.T  # [N, N]

    results = {}
    for k in k_values:
        actual_k = min(k, n)
        # TR@K
        correct = sum(
            1 for i in range(n)
            if i in sim[:, i].topk(actual_k).indices
        )
        results[f"TR@{k}"] = round(100.0 * correct / n, 2)
        # IR@K
        correct = sum(
            1 for i in range(n)
            if i in sim[i, :].topk(actual_k).indices
        )
        results[f"IR@{k}"] = round(100.0 * correct / n, 2)

    return results


# =============================================================================
# Efficiency tracker
# =============================================================================

class EfficiencyTracker:
    def __init__(self):
        self.epoch_times = []
        self.n_epochs = 0

    def start_epoch(self):
        self._t = time.time()

    def end_epoch(self):
        self.epoch_times.append(time.time() - self._t)
        self.n_epochs += 1

    def compute_cost(self):
        if not self.epoch_times:
            return {"total_time_s": 0.0, "avg_epoch_time_s": 0.0, "n_epochs": 0}
        total = sum(self.epoch_times)
        return {
            "total_time_s": round(total, 2),
            "avg_epoch_time_s": round(total / len(self.epoch_times), 2),
            "n_epochs": self.n_epochs,
        }



def compute_rbs(image_embeddings, group_labels):
    """
    Representation Bias Score (RBS).
    
    Measures how much each demographic group's centroid
    deviates from the global mean embedding.
    
    Formula: RBS = (1/K) * sum_k ||centroid_k - global_mean||^2
    
    Lower RBS = groups are more uniformly distributed = fairer.
    This is the metric your PCA/SVD directly minimizes.
    """
    valid_mask = group_labels != -1
    valid_embs = image_embeddings[valid_mask].float()
    valid_labels = group_labels[valid_mask]
    unique_groups = [g for g in valid_labels.unique() if g.item() != -1]
    
    if len(unique_groups) < 2:
        return 0.0
    
    # Global mean
    global_mean = valid_embs.mean(dim=0)
    
    # Per-group centroids
    deviations = []
    for g in unique_groups:
        mask = valid_labels == g
        centroid = valid_embs[mask].mean(dim=0)
        deviation = (centroid - global_mean).norm().item() ** 2
        deviations.append(deviation)
    
    return float(sum(deviations) / len(deviations))

# =============================================================================
# Master function: compute ALL metrics
# =============================================================================

def compute_all_metrics(
    image_embeddings,
    text_embeddings,
    demographic_labels,
    attribute,
    accuracy_history=None,
    efficiency_tracker=None,
    use_img2img=False,
    _occ_text_embs=None,
):
    """
    Compute every metric from the proposal in one call.
    Returns a dict ready for the paper table.
    """
    results = {"attribute": attribute}

    # Filter valid labels
    valid_mask = demographic_labels != -1
    valid_img = image_embeddings[valid_mask]
    valid_labels = demographic_labels[valid_mask]

    log.info(f"Valid samples: {valid_mask.sum()}/{len(demographic_labels)}")

    # --- Classification accuracy metrics ---
    # How often does image-to-text matching give the correct demographic?
    n_prompts = len(text_embeddings)
    sim_i2t = valid_img.float() @ text_embeddings.float().T
    top1_pred = sim_i2t.argmax(dim=1)

    # Clip labels to valid prompt range
    clipped_labels = valid_labels.clone()
    clipped_labels = clipped_labels.clamp(0, n_prompts - 1)

    preds_correct = (top1_pred == clipped_labels).long()
    true_ones = torch.ones(len(valid_labels), dtype=torch.long)

    results["accuracy"] = round(compute_accuracy(preds_correct, true_ones), 4)
    prf = compute_precision_recall_f1(preds_correct, true_ones)
    results.update(prf)

    if accuracy_history:
        results["convergence_epoch"] = compute_convergence_speed(accuracy_history)

    # --- Fairness metrics ---
    results["dpg"] = round(compute_dpg_full(preds_correct, valid_labels), 4)
    
    # Representation Bias Score — directly measures embedding-space bias
    # This is what PCA/SVD (Step IV) and bias removal (Step VIII) minimize
    results["rbs"] = round(compute_rbs(valid_img, valid_labels), 4)
    results["eod"] = round(compute_eod_full(preds_correct, true_ones, valid_labels), 4)

    # --- MaxSkew and NDKL (image-to-image retrieval fairness) ---
    log.info("Computing MaxSkew (image-to-image retrieval)...")
    # CRITICAL: Balance the evaluation set before computing MaxSkew
    # FairFace is imbalanced (Group 3 = 29.5%, Group 8 = 1%)
    # Without balancing, MaxSkew measures dataset imbalance, not model bias
    # Zhang et al. CVPR 2025 uses balanced evaluation sets for this reason
    unique_g = [g for g in valid_labels.unique() if g.item() != -1]
    # Find minimum group size (cap at 200 for speed)
    min_size = min(
        min((valid_labels == g).sum().item() for g in unique_g),
        200
    )
    balanced_idx = []
    for g in unique_g:
        g_idx = (valid_labels == g).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(g_idx))[:min_size]
        balanced_idx.append(g_idx[perm])
    balanced_idx = torch.cat(balanced_idx)
    sample_img = valid_img[balanced_idx]
    sample_labels = valid_labels[balanced_idx]
    log.info(f"  Balanced eval: {len(unique_g)} groups × {min_size} samples = {len(sample_img)} total")

    # Zhang-style: neutral occupation TEXT queries retrieve images, measure skew.
    # Encode occupation prompts using the SAME text encoder via passed-in embeds.
    if _occ_text_embs is not None:
        ms, nd = maxskew_ndkl_textquery(_occ_text_embs, valid_img, valid_labels,
                                        k_skew=50, k_ndkl=50)
        results["maxskew_5"] = round(ms, 4)
        results["ndkl"] = round(nd, 4)
    else:
        results["maxskew_5"] = round(compute_maxskew_correct(sample_img, sample_labels, k=5), 4)
        results["ndkl"] = round(compute_ndkl_correct(sample_img, sample_labels, k=10), 4)

    # --- ABLE ---
    log.info("Computing ABLE...")
    from evaluation.metrics.able import compute_able as _zhang_able
    able_result = _zhang_able(accuracy=results["accuracy"], maxskew=results["maxskew_5"])
    results["able"] = able_result["able"]
    results["vl_alignment"] = able_result.get("accuracy_term", results["accuracy"])
    results["bias_level"] = able_result.get("maxskew", 0.0)

    # --- Retrieval TR@K, IR@K ---
    log.info("Computing Recall@K...")
    n_eval = min(500, len(valid_img), n_prompts)
    recall = compute_recall_at_k(
        valid_img[:n_eval],
        text_embeddings[:n_eval] if len(text_embeddings) >= n_eval else text_embeddings,
        k_values=[1, 5, 10],
    )
    results.update(recall)

    # --- Efficiency ---
    if efficiency_tracker:
        results.update(efficiency_tracker.compute_cost())

    return results


# =============================================================================
# Zhang-style MaxSkew / NDKL — NEUTRAL TEXT QUERIES (not image-to-image)
# =============================================================================
# 16 neutral occupation prompts. No demographic words → any skew in the
# retrieved images reflects MODEL bias (doctor=male, nurse=female effect).
OCCUPATION_PROMPTS = [
    # --- Occupation / status bias (doctor=male, nurse=female) ---
    "a photo of a doctor", "a photo of a nurse", "a photo of an engineer",
    "a photo of a teacher", "a photo of a scientist", "a photo of a chef",
    "a photo of a lawyer", "a photo of a farmer", "a photo of a cleaner",
    "a photo of a manager", "a photo of a secretary", "a photo of a pilot",
    "a photo of a construction worker", "a photo of an artist",
    "a photo of a librarian", "a photo of a ceo",
    # --- Character / trait bias (e.g. black=criminal stereotype) ---
    "a photo of a criminal", "a photo of a trustworthy person",
    "a photo of an intelligent person", "a photo of a dangerous person",
    "a photo of a kind person", "a photo of a violent person",
    # --- Social status / rights bias (wealthy vs poor, powerful vs weak) ---
    "a photo of a wealthy person", "a photo of a poor person",
    "a photo of an educated person", "a photo of a powerful person",
    "a photo of a respected person", "a photo of a suspicious person",
]


def maxskew_ndkl_textquery(occ_text_embs, image_embs, group_labels,
                           k_skew=50, k_ndkl=50):
    """
    Zhang-style fairness: for each neutral occupation TEXT query, retrieve the
    top-k images and measure demographic skew in the retrieved set.

    occ_text_embs: [Q, D] occupation prompt embeddings
    image_embs:    [N, D] image embeddings
    group_labels:  [N] demographic group id per image
    Returns (mean_maxskew, mean_ndkl). Lower = fairer. Bounded (1/k floor).
    """
    unique = [g for g in group_labels.unique() if g.item() != -1]
    n_groups = len(unique)
    if n_groups < 2:
        return 0.0, 0.0

    # desired = population distribution of each group (fair target)
    desired = {g.item(): max((group_labels == g).float().mean().item(), 1e-6)
               for g in unique}

    sim = occ_text_embs.float() @ image_embs.float().T   # [Q, N]
    ms_list, nd_list = [], []

    for q in range(sim.shape[0]):
        ks = min(k_skew, sim.shape[1])
        topk = sim[q].topk(ks).indices
        labels_k = group_labels[topk]
        # MaxSkew: floor retrieved prob at 1/k (Geyik et al.) → no blow-up
        skews = []
        for g in unique:
            p_ret = max((labels_k == g).sum().item() / ks, 1.0 / ks)
            skews.append(abs(np.log(p_ret) - np.log(desired[g.item()])))
        ms_list.append(max(skews))

        # NDKL: rank-discounted KL of cumulative dist vs desired
        kn = min(k_ndkl, sim.shape[1])
        discount = np.array([1.0 / np.log2(i + 2) for i in range(kn)])
        Z = discount.sum()
        cum = {g.item(): 0 for g in unique}
        kl_sum = 0.0
        for pos in range(kn):
            cum[labels_k[pos].item()] += 1
            tot = pos + 1
            kl = sum((cum[g] / tot) * np.log((cum[g] / tot) / desired[g])
                     for g in cum if cum[g] > 0)
            kl_sum += discount[pos] * kl
        nd_list.append(kl_sum / Z)

    return float(np.mean(ms_list)), float(np.mean(nd_list))

# =============================================================================
# ZHANG-COMPARABLE MaxSkew/NDKL — neutral demographic queries, larger k
# Use this ONLY for direct comparison to Zhang's published numbers.
# (The 28-probe version above is our OWN comprehensive measurement.)
# =============================================================================
ZHANG_NEUTRAL_QUERIES = [
    "a photo of a person", "a photo of a human face", "a photo of someone",
    "a photo of an individual", "a portrait of a person",
    "a photo of a face", "a picture of a person", "a headshot of a person",
]

def maxskew_ndkl_zhang(neutral_text_embs, image_embs, group_labels,
                       k_skew=1000, k_ndkl=1000):
    """
    Zhang/Geyik-style MaxSkew & NDKL with neutral queries and large k,
    so values land in Zhang's 0.05-0.6 range for fair comparison.
    """
    import numpy as np
    unique = [g for g in group_labels.unique() if g.item() != -1]
    if len(unique) < 2:
        return 0.0, 0.0
    desired = {g.item(): max((group_labels == g).float().mean().item(), 1e-6)
               for g in unique}
    sim = neutral_text_embs.float() @ image_embs.float().T
    N = sim.shape[1]
    ms_list, nd_list = [], []
    for q in range(sim.shape[0]):
        ks = min(k_skew, N)
        topk = sim[q].topk(ks).indices
        labels_k = group_labels[topk]
        skews = []
        for g in unique:
            p_ret = max((labels_k == g).sum().item() / ks, 1.0 / ks)
            skews.append(abs(np.log(p_ret) - np.log(desired[g.item()])))
        ms_list.append(max(skews))
        kn = min(k_ndkl, N)
        discount = np.array([1.0 / np.log2(i + 2) for i in range(kn)])
        Z = discount.sum()
        cum = {g.item(): 0 for g in unique}
        kl_sum = 0.0
        for pos in range(kn):
            cum[labels_k[pos].item()] += 1
            tot = pos + 1
            kl = sum((cum[g]/tot) * np.log((cum[g]/tot)/desired[g])
                     for g in cum if cum[g] > 0)
            kl_sum += discount[pos] * kl
        nd_list.append(kl_sum / Z)
    return float(np.mean(ms_list)), float(np.mean(nd_list))
