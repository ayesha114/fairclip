"""
=============================================================================
FairCLIP — Adaptive Temperature Scaling (Step VII of Methodology)
=============================================================================
THIS IS YOUR NOVELTY #4 — Temperature that adjusts based on group disparity.

What this file does (in plain English):
    In CLIP's contrastive loss, there is a "temperature" parameter τ (tau).
    Temperature controls how SHARP or SMOOTH the similarity distribution is:
    - Low τ (e.g., 0.01): very sharp — model is very confident
    - High τ (e.g., 1.0): very smooth — model is less confident

    Standard CLIP uses a FIXED temperature (usually τ = 0.07).
    OR a LEARNABLE temperature (single scalar that updates during training).
    Neither of these considers FAIRNESS.

    YOUR IDEA: Make τ ADAPTIVE — increase it when demographic groups are
    very spread apart (high bias), decrease it when they're close (low bias).

    Why does this help?
        When bias is HIGH (groups far apart), we want a higher temperature
        to create a softer loss landscape, giving the model more room to
        move the groups closer together.

        When bias is LOW (groups already close), we use standard temperature
        so the model can focus on accuracy.

    This is the first adaptive temperature approach in CLIP debiasing.
    No existing paper (Zhang, BendVLM, CLIP-clip) does this.

Formula:
    τ = τ_base × (1 + α × group_disparity)

    Where:
        τ_base = base temperature (e.g., 0.07)
        α = sensitivity parameter (how much bias affects temperature)
        group_disparity = distance between group centroids in embedding space

Usage:
    controller = AdaptiveTemperatureController(tau_base=0.07, alpha=0.5)
    tau = controller.compute(image_embeddings, gender_labels)
    loss = infonce_loss(image_embs, text_embs, temperature=tau)
=============================================================================
"""

import logging
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class AdaptiveTemperatureController(nn.Module):
    """
    Computes adaptive temperature τ based on current demographic group disparity.

    This implements Step VII: "Use adaptive temperature scaling."

    The temperature adapts every training step:
    - High group disparity (much bias) → higher τ → softer loss
    - Low group disparity (less bias)  → lower τ → sharper loss

    This creates a natural curriculum:
    Early training: high bias → high temperature → gentle nudges
    Late training:  low bias  → low temperature → fine-tuning

    Args:
        tau_base: Base temperature (default 0.07, same as original CLIP)
        alpha: Sensitivity — how much disparity affects temperature
               0.0 = no adaptation (same as fixed temperature)
               0.5 = moderate adaptation (recommended)
               2.0 = strong adaptation
        tau_min: Minimum temperature (safety floor — prevents instability)
        tau_max: Maximum temperature (safety ceiling)
    """

    def __init__(
        self,
        tau_base: float = 0.07,
        alpha: float = 0.5,
        tau_min: float = 0.01,
        tau_max: float = 0.5,
    ):
        super().__init__()
        self.tau_base = tau_base
        self.alpha = alpha
        self.tau_min = tau_min
        self.tau_max = tau_max

        # Track history for logging and visualization (RO5)
        self.temperature_history = []
        self.disparity_history = []

        log.info(
            f"AdaptiveTemperatureController: "
            f"τ_base={tau_base}, α={alpha}, "
            f"range=[{tau_min}, {tau_max}]"
        )

    def compute_group_disparity(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """
        Compute the current demographic group disparity.

        Disparity = average pairwise distance between group centroids.
        High disparity means groups are far apart (much bias).
        Low disparity means groups are close (little bias).

        Args:
            embeddings: [N, D] L2-normalized embeddings
            labels: [N] demographic labels

        Returns:
            disparity: float scalar
        """
        # Find valid groups
        valid_mask = labels != -1
        if valid_mask.sum() == 0:
            return 0.0

        valid_embs = embeddings[valid_mask].detach().float()
        valid_labels = labels[valid_mask]
        unique_groups = valid_labels.unique()

        # Need at least 2 groups
        if len(unique_groups) < 2:
            return 0.0

        # Compute centroid for each group
        centroids = []
        for group in unique_groups:
            mask = valid_labels == group
            centroid = valid_embs[mask].mean(dim=0)
            centroids.append(centroid)

        # Average pairwise distance between centroids
        distances = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dist = (centroids[i] - centroids[j]).norm().item()
                distances.append(dist)

        return float(sum(distances) / len(distances)) if distances else 0.0

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """
        Compute the adaptive temperature for the current batch.

        Formula: τ = clip(τ_base × (1 + α × disparity), τ_min, τ_max)

        Args:
            embeddings: [N, D] current batch embeddings
            labels: [N] demographic labels

        Returns:
            temperature: float — the τ to use for THIS training step
        """
        # Measure current group disparity
        disparity = self.compute_group_disparity(embeddings, labels)

        # Compute adaptive temperature
        # When disparity is high → temperature increases above τ_base
        # When disparity is 0 → temperature equals τ_base
        tau = self.tau_base * (1.0 + self.alpha * disparity)

        # Clamp to safe range
        tau = max(self.tau_min, min(self.tau_max, tau))

        # Log history for visualization later (RO5)
        self.temperature_history.append(tau)
        self.disparity_history.append(disparity)

        return tau

    def get_current_stats(self) -> dict:
        """
        Return current temperature stats for logging.
        Called every N steps by the training loop.
        """
        if not self.temperature_history:
            return {"temperature": self.tau_base, "disparity": 0.0}

        return {
            "temperature_current": self.temperature_history[-1],
            "temperature_mean": sum(self.temperature_history) / len(self.temperature_history),
            "temperature_min": min(self.temperature_history),
            "temperature_max": max(self.temperature_history),
            "disparity_current": self.disparity_history[-1],
            "disparity_mean": sum(self.disparity_history) / len(self.disparity_history),
            "n_steps": len(self.temperature_history),
        }

    def reset_history(self):
        """Clear history (call at start of each epoch)."""
        self.temperature_history = []
        self.disparity_history = []
