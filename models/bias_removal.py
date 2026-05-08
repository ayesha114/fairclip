"""
=============================================================================
FairCLIP — Bias Removal from Embeddings (Step VIII of Methodology)
=============================================================================
What this file does (in plain English):
    After we have:
    - The bias subspace (from Step IV — PCA/SVD)
    - Aligned bias directions (from Step V — Procrustes)

    We can REMOVE bias from any embedding by projecting it OUT of
    the bias subspace.

    Think of it like this:
    - An embedding is a point in 512-dimensional space
    - The bias subspace is a small "room" inside that space
    - We remove the part of the embedding that "points toward" that room
    - What remains is the debiased embedding

    This is Step VIII: "Remove bias from embeddings."

    This is used in TWO ways:
    1. During training: applied to embeddings before the contrastive loss
    2. At inference time: applied to any new embedding to debias it

Usage:
    remover = BiasRemover()
    remover.set_bias_subspace(image_bias_dirs, text_bias_dirs)
    debiased_image_embs = remover.remove_image_bias(image_embs)
    debiased_text_embs  = remover.remove_text_bias(text_embs)
=============================================================================
"""

import logging
from typing import Optional
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class BiasRemover(nn.Module):
    """
    Removes bias from embeddings by projecting out the bias subspace.

    This is Step VIII of the methodology: "Remove bias from embeddings."

    The bias subspace comes from BiasSubspaceDiscoverer (Step IV) and
    ProcrustesAligner (Step V).

    After alignment, both image and text bias directions point in the
    same direction, so removing them from both modalities is consistent.
    """

    def __init__(self):
        super().__init__()
        # Set by set_bias_subspace() after Steps IV and V complete
        self.image_bias_directions: Optional[torch.Tensor] = None
        self.text_bias_directions: Optional[torch.Tensor] = None

    def set_bias_subspace(
        self,
        image_bias_directions: torch.Tensor,
        text_bias_directions: torch.Tensor,
    ) -> None:
        """
        Set the bias directions discovered in Steps IV and V.

        Call this ONCE after running BiasSubspaceDiscoverer and
        ProcrustesAligner. Then use remove_image_bias() and
        remove_text_bias() freely.

        Args:
            image_bias_directions: [D, k] from BiasSubspaceDiscoverer
            text_bias_directions: [D, k] aligned by ProcrustesAligner
        """
        self.image_bias_directions = image_bias_directions.float()
        self.text_bias_directions = text_bias_directions.float()
        log.info(
            f"Bias subspace set: "
            f"image_dirs={image_bias_directions.shape}, "
            f"text_dirs={text_bias_directions.shape}"
        )

    def _project_out(
        self,
        embeddings: torch.Tensor,
        bias_directions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Core projection math: remove the bias component from embeddings.

        For each embedding e:
            bias_component = B @ (B^T @ e)
            debiased = e - bias_component
            debiased = debiased / ||debiased||  (re-normalize)

        Where B is the [D, k] bias direction matrix.

        Args:
            embeddings: [N, D]
            bias_directions: [D, k]

        Returns:
            debiased: [N, D] L2-normalized
        """
        B = bias_directions.to(embeddings.device)
        # Project onto bias subspace
        proj = embeddings @ B          # [N, k]
        # Reconstruct bias component
        bias = proj @ B.T              # [N, D]
        # Remove it
        debiased = embeddings - bias   # [N, D]
        # Re-normalize
        norms = debiased.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return debiased / norms

    def remove_image_bias(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Remove bias from image embeddings.

        Args:
            embeddings: [N, D] image embeddings from CLIPBackbone

        Returns:
            debiased: [N, D] bias-removed image embeddings
        """
        if self.image_bias_directions is None:
            raise RuntimeError(
                "Call set_bias_subspace() before remove_image_bias()."
            )
        return self._project_out(embeddings.float(), self.image_bias_directions)

    def remove_text_bias(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Remove bias from text embeddings.

        Args:
            embeddings: [N, D] text embeddings from CLIPBackbone

        Returns:
            debiased: [N, D] bias-removed text embeddings
        """
        if self.text_bias_directions is None:
            raise RuntimeError(
                "Call set_bias_subspace() before remove_text_bias()."
            )
        return self._project_out(embeddings.float(), self.text_bias_directions)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Remove bias from both image and text embeddings in one call.

        Args:
            image_embeddings: [N, D]
            text_embeddings: [N, D]

        Returns:
            (debiased_image, debiased_text): both [N, D]
        """
        return (
            self.remove_image_bias(image_embeddings),
            self.remove_text_bias(text_embeddings),
        )
