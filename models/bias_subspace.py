"""
=============================================================================
FairCLIP — Bias Subspace Discovery (Step IV of Methodology)
=============================================================================
THIS IS YOUR NOVELTY #1 — PCA/SVD on group-wise embeddings.

What this file does (in plain English):
    Imagine all the face embeddings from CLIP laid out in space.
    Male faces cluster in one area. Female faces cluster in another.
    The LINE connecting the male cluster center to the female cluster center
    is the "gender bias direction."

    This file finds those bias directions using PCA/SVD:
    1. Compute the average embedding for each demographic group
       (e.g., average of all Male faces, average of all Female faces)
    2. Stack these group averages into a matrix
    3. Run SVD on the matrix to find the principal directions of variation
    4. The top SVD directions = the bias subspace

    We do this SEPARATELY for image embeddings and text embeddings.
    Then Step V (Procrustes) aligns them.

Why PCA/SVD instead of a learned MLP (like Zhang et al.)?
    - Interpretable: you can visualize the bias directions (RO5)
    - No training needed: computed analytically in one pass
    - Closed-form: always gives the same answer (deterministic)
    - Efficient: fast even on CPU

Reference: Bolukbasi et al. "Man is to Computer Programmer as Woman is to
           Homemaker? Debiasing Word Embeddings." NeurIPS 2016.
           (Extended here to multimodal vision-language setting)

Usage:
    # After extracting embeddings with CLIPBackbone:
    discoverer = BiasSubspaceDiscoverer(n_bias_directions=5)

    # Fit on image embeddings
    discoverer.fit_image(image_embeddings, gender_labels)

    # Get the bias directions (matrix of shape [embedding_dim, n_directions])
    image_bias_directions = discoverer.image_bias_directions

    # Project embeddings to remove bias (Step VIII)
    debiased = discoverer.remove_bias_image(image_embeddings)
=============================================================================
"""

import logging
from typing import Optional

import numpy as np
import torch
from sklearn.decomposition import PCA
from scipy.linalg import svd as scipy_svd

log = logging.getLogger(__name__)


class BiasSubspaceDiscoverer:
    """
    Discovers bias directions in CLIP embeddings using PCA/SVD.

    This implements Step IV of the proposal methodology:
    "Discover bias subspace using PCA/SVD on group-wise embeddings."

    How it works:
        1. For each demographic group (e.g., Male, Female), compute the
           CENTROID (average embedding) of all samples in that group.
        2. Stack these centroids into a matrix C of shape [n_groups, embedding_dim]
        3. Center C by subtracting the global mean
        4. Run SVD on C to get the top-k singular vectors
        5. These vectors span the "bias subspace"

    The bias subspace is a low-dimensional subspace of the embedding space
    that captures the directions along which demographic groups differ most.
    """

    def __init__(self, n_bias_directions: int = 5):
        """
        Args:
            n_bias_directions: How many bias directions to extract.
                              More directions = more bias removed, but also
                              more semantic information lost.
                              We use 5 as default (same as Hard-Debias paper).
                              This is a hyperparameter to tune during ablations.
        """
        self.n_bias_directions = n_bias_directions

        # These are filled in after .fit_image() and .fit_text() are called
        # Shape: [embedding_dim, n_bias_directions] — the bias subspace basis
        self.image_bias_directions: Optional[torch.Tensor] = None
        self.text_bias_directions: Optional[torch.Tensor] = None

        # Group centroids — useful for visualization (RO5)
        self.image_centroids: Optional[dict] = None
        self.text_centroids: Optional[dict] = None

        # Global means — needed for projection
        self.image_global_mean: Optional[torch.Tensor] = None
        self.text_global_mean: Optional[torch.Tensor] = None

        log.info(f"BiasSubspaceDiscoverer initialized with {n_bias_directions} bias directions")

    def _compute_group_centroids(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[dict, torch.Tensor]:
        """
        Compute the average embedding for each demographic group.

        This is the core of Step IV: we look at where each group
        'lives' in the embedding space.

        Args:
            embeddings: [N, D] tensor of L2-normalized embeddings
            labels: [N] tensor of integer group labels (e.g., 0=Male, 1=Female)

        Returns:
            centroids: dict mapping label -> centroid tensor [D]
            global_mean: mean of all embeddings [D]
        """
        embeddings = embeddings.float()
        unique_labels = labels.unique()
        centroids = {}

        for label in unique_labels:
            label_int = label.item()

            # Skip the -1 sentinel (missing/ambiguous labels)
            if label_int == -1:
                continue

            # Get all embeddings belonging to this group
            mask = labels == label
            group_embeddings = embeddings[mask]  # [n_group, D]

            # Centroid = mean of all group embeddings
            centroid = group_embeddings.mean(dim=0)  # [D]
            centroids[label_int] = centroid

            log.info(
                f"  Group {label_int}: {mask.sum()} samples, "
                f"centroid norm={centroid.norm():.4f}"
            )

        # Global mean (mean of all valid embeddings)
        valid_mask = labels != -1
        global_mean = embeddings[valid_mask].mean(dim=0)

        return centroids, global_mean

    def _run_svd(
        self,
        centroids: dict,
        global_mean: torch.Tensor,
        n_directions: int,
    ) -> torch.Tensor:
        """
        Run SVD on the centroid matrix to find bias directions.

        Mathematical explanation:
            Let C be the [n_groups, D] matrix of centered centroids.
            SVD gives C = U * S * V^T
            The columns of V (right singular vectors) are the bias directions.
            The top-k columns of V correspond to the k largest singular values,
            meaning they capture the MOST variance between groups.
            These are our bias directions.

        Args:
            centroids: dict of label -> centroid tensor [D]
            global_mean: global mean embedding [D]
            n_directions: how many directions to keep

        Returns:
            bias_directions: [D, n_directions] matrix
                            Each column is one bias direction (unit vector)
        """
        # Stack centroids into a matrix: [n_groups, D]
        centroid_matrix = torch.stack(list(centroids.values()), dim=0)

        # Center by subtracting global mean
        # This focuses SVD on GROUP DIFFERENCES, not absolute position
        centered = centroid_matrix - global_mean.unsqueeze(0)

        # Convert to numpy for scipy SVD (more numerically stable than torch SVD)
        C = centered.cpu().numpy()

        # Run full SVD: C = U * diag(S) * V^T
        # V has shape [D, D] — each ROW is a right singular vector
        U, S, Vt = scipy_svd(C, full_matrices=False)

        # The top-k rows of Vt are our bias directions
        # We transpose to get [D, k] where each COLUMN is a direction
        n_keep = min(n_directions, len(S), len(centroids) - 1 if len(centroids) > 1 else 1)
        bias_directions = torch.tensor(Vt[:n_keep].T, dtype=torch.float32)

        # Log the explained variance for each direction
        total_variance = (S ** 2).sum()
        for i, s in enumerate(S[:n_keep]):
            explained = 100 * (s ** 2) / total_variance
            log.info(f"  Bias direction {i+1}: singular value={s:.4f}, "
                    f"explained variance={explained:.1f}%")

        return bias_directions  # [D, n_directions]

    def fit_image(
        self,
        image_embeddings: torch.Tensor,
        labels: torch.Tensor,
        attribute: str = "gender",
    ) -> "BiasSubspaceDiscoverer":
        """
        Discover bias directions in IMAGE embeddings.

        Call this with the image embeddings from CLIPBackbone.encode_images()
        and the corresponding demographic labels.

        Args:
            image_embeddings: [N, D] L2-normalized image embeddings from CLIP
            labels: [N] integer demographic labels
                    (e.g., gender_idx: 0=Male, 1=Female)
            attribute: Name of the attribute being analyzed (for logging)

        Returns:
            self (for method chaining)
        """
        log.info(f"Fitting image bias subspace for attribute: {attribute}")
        log.info(f"  Input: {image_embeddings.shape[0]} embeddings, "
                f"dim={image_embeddings.shape[1]}")

        # Step 1: Compute group centroids
        centroids, global_mean = self._compute_group_centroids(
            image_embeddings, labels
        )

        # Step 2: Run SVD to find bias directions
        bias_directions = self._run_svd(
            centroids, global_mean, self.n_bias_directions
        )

        # Store for later use
        self.image_bias_directions = bias_directions
        self.image_centroids = centroids
        self.image_global_mean = global_mean

        log.info(f"Image bias subspace discovered: {bias_directions.shape}")
        return self

    def fit_text(
        self,
        text_embeddings: torch.Tensor,
        labels: torch.Tensor,
        attribute: str = "gender",
    ) -> "BiasSubspaceDiscoverer":
        """
        Discover bias directions in TEXT embeddings.

        Call this with text embeddings from demographic prompts like:
        ["A photo of a Male person", "A photo of a Female person", ...]

        Args:
            text_embeddings: [N, D] L2-normalized text embeddings from CLIP
            labels: [N] integer demographic labels
            attribute: Name of the attribute (for logging)

        Returns:
            self (for method chaining)
        """
        log.info(f"Fitting text bias subspace for attribute: {attribute}")

        centroids, global_mean = self._compute_group_centroids(
            text_embeddings, labels
        )

        bias_directions = self._run_svd(
            centroids, global_mean, self.n_bias_directions
        )

        self.text_bias_directions = bias_directions
        self.text_centroids = centroids
        self.text_global_mean = global_mean

        log.info(f"Text bias subspace discovered: {bias_directions.shape}")
        return self

    def remove_bias_image(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Remove bias from image embeddings (Step VIII of methodology).

        How it works:
            For each embedding, we subtract the component that lies in
            the bias subspace. This is called "null-space projection."

            Mathematically:
                debiased = embedding - B * B^T * embedding
            where B is the bias direction matrix [D, k].

            This removes exactly the part of the embedding that points
            in a "biased" direction.

        Args:
            embeddings: [N, D] image embeddings to debias

        Returns:
            debiased: [N, D] embeddings with bias removed, re-normalized
        """
        if self.image_bias_directions is None:
            raise RuntimeError(
                "Call fit_image() before remove_bias_image(). "
                "The bias subspace has not been computed yet."
            )

        return self._project_out_bias(embeddings, self.image_bias_directions)

    def remove_bias_text(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Remove bias from text embeddings (Step VIII of methodology).

        Same as remove_bias_image() but uses the text bias directions.
        """
        if self.text_bias_directions is None:
            raise RuntimeError(
                "Call fit_text() before remove_bias_text(). "
                "The bias subspace has not been computed yet."
            )

        return self._project_out_bias(embeddings, self.text_bias_directions)

    def _project_out_bias(
        self,
        embeddings: torch.Tensor,
        bias_directions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project out the bias subspace from embeddings.

        This is the mathematical heart of bias removal.
        For each embedding e:
            bias_component = B * (B^T * e)  -- component along bias directions
            debiased = e - bias_component   -- remove that component

        Args:
            embeddings: [N, D]
            bias_directions: [D, k] — the bias subspace basis

        Returns:
            debiased: [N, D], L2 re-normalized
        """
        embeddings = embeddings.float()
        B = bias_directions.to(embeddings.device)  # [D, k]

        # Project onto bias subspace: [N, k]
        projections = embeddings @ B  # [N, k]

        # Reconstruct the bias component: [N, D]
        bias_component = projections @ B.T  # [N, D]

        # Remove the bias component
        debiased = embeddings - bias_component  # [N, D]

        # Re-normalize to unit sphere (CLIP embeddings should be unit vectors)
        norms = debiased.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        debiased = debiased / norms

        return debiased

    def get_bias_score(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        modality: str = "image",
    ) -> float:
        """
        Compute a scalar bias score for the given embeddings.

        This measures HOW MUCH bias is present by computing the average
        distance between group centroids in the bias subspace.
        Higher score = more bias.

        Used for logging and evaluation (Step IX).

        Args:
            embeddings: [N, D]
            labels: [N] integer labels
            modality: "image" or "text"

        Returns:
            bias_score: float (higher = more bias)
        """
        directions = (
            self.image_bias_directions
            if modality == "image"
            else self.text_bias_directions
        )

        if directions is None:
            return float("nan")

        B = directions.to(embeddings.device)

        # Project embeddings onto bias subspace
        projected = embeddings @ B  # [N, k]

        # Compute per-group means in bias subspace
        unique_labels = labels.unique()
        valid_labels = [l.item() for l in unique_labels if l.item() != -1]

        if len(valid_labels) < 2:
            return 0.0

        group_means = []
        for label in valid_labels:
            mask = labels == label
            group_mean = projected[mask].mean(dim=0)
            group_means.append(group_mean)

        # Bias score = average pairwise distance between group means
        # in the bias subspace
        distances = []
        for i in range(len(group_means)):
            for j in range(i + 1, len(group_means)):
                dist = (group_means[i] - group_means[j]).norm().item()
                distances.append(dist)

        return float(np.mean(distances)) if distances else 0.0

    def save(self, path: str) -> None:
        """Save the fitted bias subspace to disk for reuse."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "image_bias_directions": self.image_bias_directions,
            "text_bias_directions": self.text_bias_directions,
            "image_global_mean": self.image_global_mean,
            "text_global_mean": self.text_global_mean,
            "n_bias_directions": self.n_bias_directions,
        }, path)
        log.info(f"Saved bias subspace to {path}")

    def load(self, path: str) -> "BiasSubspaceDiscoverer":
        """Load a previously fitted bias subspace from disk."""
        checkpoint = torch.load(path, map_location="cpu")
        self.image_bias_directions = checkpoint["image_bias_directions"]
        self.text_bias_directions = checkpoint["text_bias_directions"]
        self.image_global_mean = checkpoint["image_global_mean"]
        self.text_global_mean = checkpoint["text_global_mean"]
        self.n_bias_directions = checkpoint["n_bias_directions"]
        log.info(f"Loaded bias subspace from {path}")
        return self
