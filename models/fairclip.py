"""
=============================================================================
FairCLIP — Main Model (Wires Steps III–VIII Together)
=============================================================================
This is the CENTRAL file of your thesis implementation.

It connects ALL your components in the correct order:

    Input (images + text + demographic labels)
        ↓
    Step III:  CLIPBackbone → image embeddings + text embeddings
        ↓
    Step IV:   BiasSubspaceDiscoverer → discover bias directions
        ↓
    Step V:    ProcrustesAligner → align text bias to image bias
        ↓
    Step VI:   FairnessAwareLoss → compute combined training loss
        ↓
    Step VII:  AdaptiveTemperatureController → compute τ for this batch
        ↓
    Step VIII: BiasRemover → project out bias from embeddings
        ↓
    Output: debiased image + text embeddings + training loss

Usage:
    model = FairCLIP(model_name="ViT-B/32", device="cuda")
    model.fit_bias_subspace(image_embs, text_embs, labels)
    loss, components = model.training_step(images, texts, labels)
    debiased_img, debiased_txt = model.debias(image_embs, text_embs)
=============================================================================
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn

from models.clip_backbone import CLIPBackbone
from models.bias_subspace import BiasSubspaceDiscoverer
from models.procrustes_alignment import ProcrustesAligner
from models.adaptive_temperature import AdaptiveTemperatureController
from models.bias_removal import BiasRemover
from losses.fairness_regularizer import FairnessAwareLoss

log = logging.getLogger(__name__)


class FairCLIP(nn.Module):
    """
    FairCLIP: Training-time joint vision-language debiasing.

    This is the complete implementation of your proposed methodology.
    All 9 steps of the proposal are implemented here.

    Two-phase operation:
        Phase 1 — Bias subspace fitting (Steps III, IV, V):
            Run once on a representative sample of the training data.
            Discovers and aligns bias directions.
            Call: model.fit_bias_subspace(image_embs, text_embs, labels)

        Phase 2 — Fair contrastive training (Steps VI, VII, VIII):
            Run every training step.
            Applies fairness-aware loss with adaptive temperature.
            Call: loss, info = model.training_step(images, texts, labels)
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = "cuda",
        n_bias_directions: int = 5,
        lambda_fair_image: float = 0.1,
        lambda_fair_text: float = 0.1,
        tau_base: float = 0.07,
        alpha_temperature: float = 0.5,
    ):
        """
        Args:
            model_name: CLIP backbone ("ViT-B/32" or "ViT-B/16")
            device: "cuda" or "cpu"
            n_bias_directions: number of PCA bias directions (Step IV)
            lambda_fair_image: fairness loss weight for images (Step VI)
            lambda_fair_text: fairness loss weight for text (Step VI)
            tau_base: base temperature (Step VII)
            alpha_temperature: how much disparity affects temperature (Step VII)
        """
        super().__init__()

        self.device = device
        self.model_name = model_name

        log.info("=" * 60)
        log.info("Initializing FairCLIP")
        log.info(f"  Backbone: {model_name}")
        log.info(f"  Bias directions: {n_bias_directions}")
        log.info(f"  Lambda fair: img={lambda_fair_image}, txt={lambda_fair_text}")
        log.info(f"  Temperature: base={tau_base}, alpha={alpha_temperature}")
        log.info("=" * 60)

        # Step III — CLIP backbone (frozen initially)
        self.backbone = CLIPBackbone(
            model_name=model_name,
            device=device,
            freeze_backbone=True,
        )

        # Step IV — Bias subspace discoverer
        self.bias_discoverer = BiasSubspaceDiscoverer(
            n_bias_directions=n_bias_directions
        )

        # Step V — Procrustes aligner
        self.procrustes = ProcrustesAligner()

        # Step VI — Fairness-aware contrastive loss
        self.loss_fn = FairnessAwareLoss(
            lambda_fair_image=lambda_fair_image,
            lambda_fair_text=lambda_fair_text,
        )

        # Step VII — Adaptive temperature controller
        self.temperature_controller = AdaptiveTemperatureController(
            tau_base=tau_base,
            alpha=alpha_temperature,
        )

        # Step VIII — Bias remover
        self.bias_remover = BiasRemover()

        # Track whether bias subspace has been fitted
        self._bias_subspace_fitted = False

        self.to(device)

    # =========================================================================
    # Phase 1: Bias Subspace Fitting (Steps IV + V)
    # =========================================================================

    def fit_bias_subspace(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        labels: torch.Tensor,
        attribute: str = "gender",
    ) -> None:
        """
        Fit the bias subspace on a representative sample. (Steps IV + V)

        Call this ONCE before training begins.
        Typically run on the full training set or a large sample.

        What happens:
            1. Run PCA/SVD on image embeddings grouped by demographic label
            2. Run PCA/SVD on text embeddings grouped by demographic label
            3. Run Procrustes to align text bias directions to image bias
            4. Set the aligned directions in the BiasRemover

        Args:
            image_embeddings: [N, D] image embeddings from CLIP
            text_embeddings: [N, D] text embeddings from CLIP
            labels: [N] demographic labels (gender_idx, age_idx, or race_idx)
            attribute: which attribute we're debiasing (for logging)
        """
        log.info(f"Fitting bias subspace for attribute: {attribute}")

        # Step IV: Discover bias subspace in each modality separately
        self.bias_discoverer.fit_image(image_embeddings, labels, attribute)
        text_labels = torch.arange(len(text_embeddings))
        self.bias_discoverer.fit_text(text_embeddings, text_labels, attribute)

        img_dirs = self.bias_discoverer.image_bias_directions
        txt_dirs = self.bias_discoverer.text_bias_directions

        # Match dimensions — trim to minimum
        min_dirs = min(img_dirs.shape[1], txt_dirs.shape[1])
        img_dirs = img_dirs[:, :min_dirs]
        txt_dirs = txt_dirs[:, :min_dirs]
        self.bias_discoverer.image_bias_directions = img_dirs
        self.bias_discoverer.text_bias_directions = txt_dirs

        # Step V: Align text bias directions to image bias directions
        aligned_txt_dirs = self.procrustes.fit_transform(img_dirs, txt_dirs)

        log.info(
            f"Procrustes alignment quality: "
            f"{self.procrustes.get_alignment_quality()}"
        )

        # Set aligned directions in the bias remover (Step VIII)
        self.bias_remover.set_bias_subspace(img_dirs, aligned_txt_dirs)

        self._bias_subspace_fitted = True
        log.info("Bias subspace fitted and ready for training")

    # =========================================================================
    # Phase 2: Training Step (Steps VI + VII + VIII)
    # =========================================================================

    def training_step(
        self,
        images: torch.Tensor,
        texts: list[str],
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        One training step — runs Steps III, VI, VII, VIII.

        Called by the training loop for every batch.

        What happens:
            1. Encode images and texts with CLIP (Step III)
            2. Compute adaptive temperature based on current bias (Step VII)
            3. Apply fairness-aware contrastive loss (Step VI)
            4. Return loss for backpropagation

        Note: Bias removal (Step VIII) is done AFTER loss computation
              because we want the loss to push toward unbiased embeddings,
              not just mask the bias.

        Args:
            images: [N, 3, H, W] preprocessed image tensors
            texts: list of N text strings (captions)
            labels: [N] demographic labels for fairness penalty

        Returns:
            loss: scalar tensor (call .backward() on this)
            info: dict with individual loss components for logging
        """
        if not self._bias_subspace_fitted:
            raise RuntimeError(
                "Call fit_bias_subspace() before training_step(). "
                "The bias subspace must be fitted on the training data first."
            )

        # Step III: Extract features using CLIP (with gradients)
        image_embs = self.backbone.encode_images_grad(images)
        text_embs = self.backbone.encode_text_grad(texts)

        # Step VII: Compute adaptive temperature for this batch
        # Uses current group disparity to scale τ
        tau = self.temperature_controller(image_embs.detach(), labels)

        # Step VI: Compute fairness-aware contrastive loss
        # Loss = InfoNCE(τ) + λ_img × GroupVariance(images) + λ_txt × GroupVariance(texts)
        loss, loss_components = self.loss_fn(
            image_embs, text_embs, labels, temperature=tau
        )

        # Add temperature to the logging dict
        loss_components["temperature"] = tau

        return loss, loss_components

    # =========================================================================
    # Inference: Debias embeddings (Step VIII)
    # =========================================================================

    @torch.no_grad()
    def encode_and_debias(
        self,
        images: torch.Tensor = None,
        texts: list[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images/texts and return DEBIASED embeddings.

        Used at evaluation time (Step IX) and for retrieval tasks.

        Args:
            images: [N, 3, H, W] image tensors (optional)
            texts: list of text strings (optional)

        Returns:
            (debiased_image_embs, debiased_text_embs)
            Either can be None if the input was None.
        """
        debiased_img = None
        debiased_txt = None

        if images is not None:
            img_embs = self.backbone.encode_images(images)
            debiased_img = self.bias_remover.remove_image_bias(img_embs)

        if texts is not None:
            txt_embs = self.backbone.encode_text(texts)
            debiased_txt = self.bias_remover.remove_text_bias(txt_embs)

        return debiased_img, debiased_txt

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Return raw (non-debiased) image embeddings. Used for comparison."""
        return self.backbone.encode_images(images)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Return raw (non-debiased) text embeddings. Used for comparison."""
        return self.backbone.encode_text(texts)

    def get_embedding_dim(self) -> int:
        return self.backbone.get_embedding_dim()

    # =========================================================================
    # Save / Load
    # =========================================================================

    def save(self, path: str) -> None:
        """Save the full FairCLIP model state."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_name": self.model_name,
            "bias_discoverer_image": self.bias_discoverer.image_bias_directions,
            "bias_discoverer_text": self.bias_discoverer.text_bias_directions,
            "procrustes_rotation": self.procrustes.rotation_matrix,
            "bias_fitted": self._bias_subspace_fitted,
        }, path)
        log.info(f"Saved FairCLIP to {path}")

    def load(self, path: str) -> "FairCLIP":
        """Load a saved FairCLIP model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.bias_discoverer.image_bias_directions = checkpoint["bias_discoverer_image"]
        self.bias_discoverer.text_bias_directions = checkpoint["bias_discoverer_text"]
        self.procrustes.rotation_matrix = checkpoint["procrustes_rotation"]
        self._bias_subspace_fitted = checkpoint["bias_fitted"]

        if self._bias_subspace_fitted:
            self.bias_remover.set_bias_subspace(
                self.bias_discoverer.image_bias_directions,
                self.bias_discoverer.text_bias_directions,
            )
        log.info(f"Loaded FairCLIP from {path}")
        return self

    def __repr__(self):
        return (
            f"FairCLIP(\n"
            f"  backbone={self.model_name},\n"
            f"  embedding_dim={self.get_embedding_dim()},\n"
            f"  bias_fitted={self._bias_subspace_fitted}\n"
            f")"
        )
