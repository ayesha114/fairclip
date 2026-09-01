"""
=============================================================================
FairCLIP — Fairness-Aware Contrastive Loss (Step VI of Methodology)
=============================================================================
THIS IS YOUR NOVELTY #3 — Fairness loss baked INTO training.

What this file does (in plain English):
    Standard CLIP training uses InfoNCE loss which only cares about matching
    each image to its correct text caption. It does NOT care whether the
    model treats demographic groups fairly.

    This file adds a FAIRNESS PENALTY to the loss:
        Total Loss = CLIP Loss + λ × Fairness Penalty

    The fairness penalty measures how SPREAD OUT the different demographic
    groups are in the embedding space. If male and female faces cluster in
    very different places, the penalty is HIGH. If they overlap nicely,
    the penalty is LOW.

    By minimizing this combined loss during training, the model is forced
    to BOTH match images to captions (accuracy) AND keep demographic groups
    close together (fairness).

    This directly addresses your research gap:
    "Fairness is rarely considered during training"

    And answers RQ2: "Can adding fairness rules during training reduce bias?"

Two loss components:
    1. InfoNCE (standard CLIP contrastive loss)
       - Makes matched image-text pairs similar
       - Makes unmatched pairs dissimilar

    2. Group Variance Penalty (your fairness regularizer)
       - Measures variance of group centroids
       - High variance = groups are far apart = more bias
       - Penalizing this pushes groups closer together

Usage:
    loss_fn = FairnessAwareLoss(lambda_fair=0.1)
    loss = loss_fn(image_embs, text_embs, gender_labels, temperature=0.07)
=============================================================================
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class InfoNCELoss(nn.Module):
    """
    Standard CLIP contrastive loss (InfoNCE).

    This is the original CLIP loss. For a batch of N (image, text) pairs:
    - Image i should be most similar to text i (positive pair)
    - Image i should be dissimilar to all other texts (negative pairs)
    - Same logic applies from text side too

    The loss is symmetric: computed from image→text AND text→image sides.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss.

        Args:
            image_embeddings: [N, D] L2-normalized image embeddings
            text_embeddings: [N, D] L2-normalized text embeddings
            temperature: scaling factor τ (lower = sharper distribution)
                        Standard CLIP uses 0.07

        Returns:
            loss: scalar tensor
        """
        N = image_embeddings.shape[0]

        # Similarity matrix: [N, N]
        # sim[i,j] = cosine similarity between image i and text j
        # Since embeddings are L2-normalized, dot product = cosine similarity
        sim_matrix = image_embeddings @ text_embeddings.T  # [N, N]
        sim_matrix = sim_matrix / temperature

        # Labels: diagonal elements are the positive pairs
        # image 0 matches text 0, image 1 matches text 1, etc.
        labels = torch.arange(N, device=image_embeddings.device)

        # Image-to-text loss: for each image, find its matching text
        loss_i2t = F.cross_entropy(sim_matrix, labels)

        # Text-to-image loss: for each text, find its matching image
        loss_t2i = F.cross_entropy(sim_matrix.T, labels)

        # Final loss is average of both directions (symmetric)
        loss = (loss_i2t + loss_t2i) / 2.0
        return loss


class GroupVariancePenalty(nn.Module):
    """
    Fairness regularizer: penalizes variance between demographic group centroids.

    In plain English:
        1. Compute the average embedding for each demographic group
           (e.g., average of all Male embeddings, average of all Female embeddings)
        2. Measure how SPREAD OUT these group averages are
        3. Return this spread as a penalty (higher spread = more bias = higher loss)

    By adding this to the training loss, we push the model to produce
    embeddings where demographic groups are NOT spread far apart.

    Mathematical formula:
        For groups g1, g2, ..., gK with centroids c1, c2, ..., cK:
        global_mean = mean of all centroids
        variance = (1/K) * sum_k ||c_k - global_mean||^2

    This is the BETWEEN-GROUP variance. Minimizing it reduces
    the demographic separation in the embedding space.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute group variance penalty.

        Args:
            embeddings: [N, D] L2-normalized embeddings
            labels: [N] integer demographic group labels
                    Use -1 for missing/ambiguous labels (they are ignored)

        Returns:
            penalty: scalar tensor (higher = more group separation = more bias)
        """
        # Find valid groups (ignore -1 sentinel labels)
        valid_mask = labels != -1
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        valid_embeddings = embeddings[valid_mask]
        valid_labels = labels[valid_mask]
        unique_groups = valid_labels.unique()

        # Need at least 2 groups to compute between-group variance
        if len(unique_groups) < 2:
            return torch.tensor(0.0, device=embeddings.device)

        # Compute centroid for each group
        centroids = []
        for group in unique_groups:
            group_mask = valid_labels == group
            group_embeddings = valid_embeddings[group_mask]
            centroid = group_embeddings.mean(dim=0)  # [D]
            centroids.append(centroid)

        # Stack into matrix: [K, D] where K = number of groups
        centroid_matrix = torch.stack(centroids, dim=0)

        # Global mean of all centroids: [D]
        global_mean = centroid_matrix.mean(dim=0)

        # Between-group variance:
        # For each group centroid, compute squared distance to global mean
        # Then average across groups
        deviations = centroid_matrix - global_mean.unsqueeze(0)  # [K, D]
        squared_distances = (deviations ** 2).sum(dim=-1)  # [K]
        variance = squared_distances.mean()  # scalar

        return variance



class RetrievalSkewPenalty(nn.Module):
    """
    Retrieval-fairness penalty (targets MaxSkew during training).

    For each text query, compute the soft retrieval distribution over images
    (softmax of query-image similarities). Then compute, per demographic group,
    the total retrieved probability mass. A fair retrieval gives each group a
    mass equal to its population proportion. We penalize the squared deviation
    from population proportion (a differentiable surrogate for MaxSkew/NDKL).
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_embeddings, text_embeddings, labels):
        valid = labels != -1
        if valid.sum() == 0:
            return torch.tensor(0.0, device=image_embeddings.device)
        img = image_embeddings[valid]
        lab = labels[valid]
        groups = lab.unique()
        if len(groups) < 2:
            return torch.tensor(0.0, device=image_embeddings.device)
        # population proportion per group
        pop = torch.stack([(lab == g).float().mean() for g in groups])  # [K]
        # soft retrieval distribution: for each text query -> over images
        sims = (text_embeddings[valid] @ img.T) / self.temperature  # [N, N]
        attn = torch.softmax(sims, dim=1)  # each query's mass over images [N,N]
        # retrieved mass per group, averaged over queries
        masses = []
        for g in groups:
            gmask = (lab == g).float()  # [N]
            masses.append((attn * gmask.unsqueeze(0)).sum(dim=1).mean())
        retrieved = torch.stack(masses)  # [K]
        retrieved = retrieved / retrieved.sum().clamp(min=1e-8)
        # squared deviation from population proportion
        penalty = ((retrieved - pop) ** 2).sum()
        return penalty


class FairnessAwareLoss(nn.Module):
    """
    Combined training loss for FairCLIP.

    Total Loss = InfoNCE Loss + λ_img × Image Fairness Penalty
                              + λ_txt × Text Fairness Penalty

    The λ (lambda) hyperparameters control the trade-off:
    - λ = 0: pure CLIP training (no fairness)
    - λ = 0.1: slight fairness pressure (recommended starting point)
    - λ = 1.0: strong fairness pressure (may hurt accuracy)

    This answers your RQ4: "Can the model still perform well after debiasing?"
    The key is finding the right λ — ablation studies will tune this.
    """

    def __init__(
        self,
        lambda_fair_image: float = 0.1,
        lambda_fair_text: float = 0.1,
        lambda_retrieval: float = 0.0,
    ):
        """
        Args:
            lambda_fair_image: Weight for image fairness penalty.
                              How much we penalize demographic separation
                              in the IMAGE embedding space.
            lambda_fair_text: Weight for text fairness penalty.
                             How much we penalize demographic separation
                             in the TEXT embedding space.
        """
        super().__init__()
        self.lambda_fair_image = lambda_fair_image
        self.lambda_fair_text = lambda_fair_text

        # Sub-components
        self.infonce = InfoNCELoss()
        self.group_variance = GroupVariancePenalty()
        self.lambda_retrieval = lambda_retrieval
        self.retrieval_skew = RetrievalSkewPenalty()

        log.info(
            f"FairnessAwareLoss initialized: "
            f"λ_img={lambda_fair_image}, λ_txt={lambda_fair_text}"
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        demographic_labels: torch.Tensor,
        temperature: float = 0.07,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute the combined fairness-aware loss.

        Args:
            image_embeddings: [N, D] L2-normalized image embeddings from CLIP
            text_embeddings: [N, D] L2-normalized text embeddings from CLIP
            demographic_labels: [N] integer labels (gender_idx, age_idx, or race_idx)
            temperature: τ for InfoNCE (adaptive temperature from Step VII)

        Returns:
            total_loss: scalar tensor (what we minimize)
            loss_components: dict with individual loss values for logging
                {
                    "infonce": float,
                    "fairness_image": float,
                    "fairness_text": float,
                    "total": float
                }
        """
        # Component 1: Standard CLIP contrastive loss
        loss_clip = self.infonce(image_embeddings, text_embeddings, temperature)

        # Component 2: Image fairness penalty
        # Penalizes demographic group separation in IMAGE embedding space
        loss_fair_img = self.group_variance(image_embeddings, demographic_labels)

        # Component 3: Text fairness penalty
        # Penalizes demographic group separation in TEXT embedding space
        loss_fair_txt = self.group_variance(text_embeddings, demographic_labels)

        # Combine: weighted sum
        loss_retrieval = (self.retrieval_skew(image_embeddings, text_embeddings, demographic_labels)
                          if self.lambda_retrieval > 0 else torch.tensor(0.0, device=image_embeddings.device))
        total_loss = (
            loss_clip
            + self.lambda_fair_image * loss_fair_img
            + self.lambda_fair_text * loss_fair_txt
            + self.lambda_retrieval * loss_retrieval
        )

        # Return individual components for logging and monitoring
        loss_components = {
            "infonce": loss_clip.item(),
            "fairness_image": loss_fair_img.item(),
            "fairness_text": loss_fair_txt.item(),
            "retrieval_skew": loss_retrieval.item(),
            "total": total_loss.item(),
        }

        return total_loss, loss_components
