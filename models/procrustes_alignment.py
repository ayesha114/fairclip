"""
=============================================================================
FairCLIP — Cross-Modal Bias Alignment via Orthogonal Procrustes (Step V)
=============================================================================
THIS IS YOUR NOVELTY #2 — Closed-form Procrustes alignment.

What this file does (in plain English):
    After Step IV, we have:
    - Image bias directions (how bias looks in image space)
    - Text bias directions (how bias looks in text space)

    The problem: these two bias subspaces are NOT aligned.
    A "male doctor" image might have its bias pointing in direction [0.3, 0.7, ...]
    but the text "male doctor" might have its bias pointing in [0.8, 0.1, ...]
    They're capturing the same bias but in different directions in space.

    This misalignment is WHY most debiasing methods fail — they debias one
    modality but don't fix the cross-modal interaction (your research gap!).

    Procrustes alignment solves this by finding the ROTATION MATRIX R that
    rotates the text bias directions to match the image bias directions.
    After alignment, both modalities agree on what "bias direction" means.

Why Procrustes (not a learned module like Zhang et al.)?
    - Closed-form solution (no training, no gradient needed)
    - Unique optimal solution (guaranteed to find the best rotation)
    - Interpretable: R is just a rotation, no information is lost
    - Fast: one SVD call

Mathematical background:
    Given:
        A = image bias directions [D, k]
        B = text bias directions [D, k]

    Find rotation R [k, k] that minimizes ||A - B*R||_F
    (Frobenius norm = sum of squared differences)

    Solution (Kabsch algorithm / Orthogonal Procrustes):
        1. Compute M = A^T * B  [k, k]
        2. SVD: M = U * S * V^T
        3. R = V * U^T
        4. Aligned B = B * R

    This is the SAME algorithm used in:
    - Protein structure alignment in bioinformatics
    - Point cloud alignment in 3D vision
    We apply it to bias subspace alignment — a novel use.

Usage:
    aligner = ProcrustesAligner()
    R = aligner.fit(image_bias_directions, text_bias_directions)
    aligned_text_directions = aligner.transform(text_bias_directions)
=============================================================================
"""

import logging
from typing import Optional

import torch
import numpy as np
from scipy.linalg import svd as scipy_svd

log = logging.getLogger(__name__)


class ProcrustesAligner:
    """
    Aligns text bias directions to image bias directions using the
    Orthogonal Procrustes algorithm.

    This implements Step V of the proposal: "Align bias directions
    across modalities."

    After alignment:
    - Image bias directions stay the same (reference frame)
    - Text bias directions are rotated to match image bias directions
    - Both modalities now share the same bias subspace orientation

    This is the key step that enables JOINT debiasing (answering RQ3).
    """

    def __init__(self):
        # The rotation matrix R is computed by fit() and stored here
        # Shape: [k, k] where k = n_bias_directions
        self.rotation_matrix: Optional[torch.Tensor] = None

        # Alignment quality score (0 = perfect, higher = worse alignment)
        self.alignment_error: Optional[float] = None

        # Store the reference (image) and source (text) directions for analysis
        self.reference_directions: Optional[torch.Tensor] = None
        self.source_directions: Optional[torch.Tensor] = None

    def fit(
        self,
        reference: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the optimal rotation matrix R that aligns 'source' to 'reference'.

        In our case:
            reference = image bias directions [D, k]
            source    = text bias directions  [D, k]
            R         = rotation matrix [k, k]

        After applying R: aligned_source = source @ R
        This minimizes ||reference - source @ R||_F

        Args:
            reference: [D, k] image bias directions (stays fixed)
            source: [D, k] text bias directions (will be rotated)

        Returns:
            rotation_matrix R: [k, k] orthogonal rotation matrix
        """
        log.info("Computing Procrustes alignment...")
        log.info(f"  Reference shape: {reference.shape}")
        log.info(f"  Source shape: {source.shape}")

        # Validate shapes
        if reference.shape != source.shape:
            raise ValueError(
                f"Reference and source must have the same shape. "
                f"Got reference={reference.shape}, source={source.shape}"
            )

        D, k = reference.shape

        # Step 1: Compute the cross-covariance matrix M = reference^T @ source
        # M has shape [k, k]
        # M[i,j] = dot product of reference direction i with source direction j
        # High M[i,j] means reference direction i is similar to source direction j
        M = source.T.float() @ reference.float()  # [k, k] (source->reference)

        # Step 2: SVD decomposition of M
        # M = U * diag(S) * V^T
        # U: [k, k], S: [k], V^T: [k, k]
        M_numpy = M.cpu().numpy()
        U, S, Vt = scipy_svd(M_numpy)

        # Step 3: Compute rotation matrix R = V * U^T
        # This is the Kabsch algorithm step
        # R is guaranteed to be orthogonal (R^T R = I)
        R = U @ Vt  # [k, k] Kabsch: source @ R aligns to reference

        # Handle reflection: if det(R) = -1, we have a reflection, not a rotation
        # We want a pure rotation, so flip the sign if needed
        det_R = np.linalg.det(R)
        if det_R < 0:
            log.warning(
                f"Procrustes found reflection (det={det_R:.4f}), "
                f"correcting to pure rotation"
            )
            # Flip sign of last column of U for pure rotation
            U[:, -1] *= -1
            R = U @ Vt
            det_R = np.linalg.det(R)
            log.info(f"Corrected det(R) = {det_R:.4f}")

        # Convert back to tensor
        R_tensor = torch.tensor(R, dtype=torch.float32)

        # Step 4: Compute alignment error (quality metric)
        # Lower = better alignment
        aligned_source = source.float() @ R_tensor.to(source.device)
        error = (reference.float() - aligned_source).norm(dim=0).mean().item()
        self.alignment_error = error

        # Log alignment quality
        log.info(f"Alignment complete:")
        log.info(f"  Rotation matrix shape: {R_tensor.shape}")
        log.info(f"  Alignment error: {error:.6f} (lower is better)")
        log.info(f"  det(R) = {det_R:.4f} (should be +1.0 for pure rotation)")

        # Log how well each direction aligns
        for i in range(k):
            cosine_sim = (
                reference[:, i].float().dot(aligned_source[:, i])
                / (reference[:, i].norm() * aligned_source[:, i].norm())
            ).item()
            log.info(f"  Direction {i+1} cosine similarity after alignment: {cosine_sim:.4f}")

        # Store everything
        self.rotation_matrix = R_tensor
        self.reference_directions = reference
        self.source_directions = source

        return R_tensor

    def transform(self, source_directions: torch.Tensor) -> torch.Tensor:
        """
        Apply the computed rotation to align source directions.

        Call fit() first to compute the rotation matrix.

        Args:
            source_directions: [D, k] text bias directions to align

        Returns:
            aligned_directions: [D, k] text bias directions after rotation
        """
        if self.rotation_matrix is None:
            raise RuntimeError(
                "Call fit() before transform(). "
                "The rotation matrix has not been computed yet."
            )

        # Apply rotation: aligned = source @ R
        # This rotates the text bias directions to match the image reference frame
        R = self.rotation_matrix.to(source_directions.device)
        aligned = source_directions.float() @ R

        # Re-normalize columns to unit vectors (rotation should preserve norms,
        # but we normalize for numerical stability)
        norms = aligned.norm(dim=0, keepdim=True).clamp(min=1e-8)
        aligned = aligned / norms

        return aligned

    def fit_transform(
        self,
        reference: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convenience method: fit and transform in one call.

        Args:
            reference: [D, k] image bias directions
            source: [D, k] text bias directions

        Returns:
            aligned_source: [D, k] aligned text bias directions
        """
        self.fit(reference, source)
        return self.transform(source)

    def get_alignment_quality(self) -> dict:
        """
        Return a dict of alignment quality metrics.
        Used for logging, paper tables, and ablation studies.
        """
        if self.rotation_matrix is None:
            return {"error": "Not fitted yet"}

        return {
            "alignment_error": self.alignment_error,
            "rotation_det": float(
                torch.det(self.rotation_matrix).item()
            ),
            "is_orthogonal": bool(
                torch.allclose(
                    self.rotation_matrix @ self.rotation_matrix.T,
                    torch.eye(self.rotation_matrix.shape[0]),
                    atol=1e-5
                )
            ),
        }

    def save(self, path: str) -> None:
        """Save the fitted rotation matrix to disk."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "rotation_matrix": self.rotation_matrix,
            "alignment_error": self.alignment_error,
        }, path)
        log.info(f"Saved Procrustes rotation to {path}")

    def load(self, path: str) -> "ProcrustesAligner":
        """Load a previously fitted rotation matrix from disk."""
        checkpoint = torch.load(path, map_location="cpu")
        self.rotation_matrix = checkpoint["rotation_matrix"]
        self.alignment_error = checkpoint["alignment_error"]
        log.info(f"Loaded Procrustes rotation from {path}")
        return self

    def __repr__(self):
        if self.rotation_matrix is None:
            return "ProcrustesAligner(not fitted)"
        return (
            f"ProcrustesAligner("
            f"k={self.rotation_matrix.shape[0]}, "
            f"alignment_error={self.alignment_error:.6f})"
        )
