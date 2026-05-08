"""
=============================================================================
Unit Tests for Bias Subspace (Step IV) and Procrustes Alignment (Step V)
=============================================================================
These tests verify your two novel components work correctly.
They use synthetic data — no GPU or datasets needed to run them.

Run with:
    pytest tests/test_bias_subspace.py -v
=============================================================================
"""

import pytest
import torch
import numpy as np

from models.bias_subspace import BiasSubspaceDiscoverer
from models.procrustes_alignment import ProcrustesAligner


# =============================================================================
# Helpers — create synthetic embeddings with known bias
# =============================================================================

def make_biased_embeddings(n_per_group=50, dim=512, bias_strength=0.5, seed=42):
    """
    Create synthetic embeddings where group 0 and group 1 differ along
    a known direction. Used to verify bias detection works.

    Args:
        n_per_group: samples per group
        dim: embedding dimension
        bias_strength: how strong the group difference is (0=no bias, 1=strong)
        seed: random seed

    Returns:
        embeddings: [2*n_per_group, dim] tensor
        labels: [2*n_per_group] tensor (0 or 1)
        true_bias_direction: [dim] the direction we injected bias along
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create a known bias direction (unit vector in dim-dimensional space)
    true_bias_dir = torch.zeros(dim)
    true_bias_dir[0] = 1.0  # bias is purely along dimension 0

    # Group 0: embeddings shifted in +bias direction
    group0 = torch.randn(n_per_group, dim) * 0.1
    group0 += bias_strength * true_bias_dir.unsqueeze(0)

    # Group 1: embeddings shifted in -bias direction
    group1 = torch.randn(n_per_group, dim) * 0.1
    group1 -= bias_strength * true_bias_dir.unsqueeze(0)

    # Combine and L2-normalize (as CLIP would produce)
    embeddings = torch.cat([group0, group1], dim=0)
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

    labels = torch.cat([
        torch.zeros(n_per_group, dtype=torch.long),
        torch.ones(n_per_group, dtype=torch.long),
    ])

    return embeddings, labels, true_bias_dir


# =============================================================================
# Tests: BiasSubspaceDiscoverer
# =============================================================================

class TestBiasSubspaceDiscoverer:

    def test_fit_image_runs_without_error(self):
        """Basic smoke test — fit should not crash."""
        embeddings, labels, _ = make_biased_embeddings()
        discoverer = BiasSubspaceDiscoverer(n_bias_directions=3)
        discoverer.fit_image(embeddings, labels)
        assert discoverer.image_bias_directions is not None

    def test_fit_text_runs_without_error(self):
        """Text fitting should work the same as image fitting."""
        embeddings, labels, _ = make_biased_embeddings()
        discoverer = BiasSubspaceDiscoverer(n_bias_directions=3)
        discoverer.fit_text(embeddings, labels)
        assert discoverer.text_bias_directions is not None

    def test_bias_directions_shape(self):
        """Bias directions should have shape [D, actual_directions].
        With 2 groups, SVD gives at most 1 direction (n_groups - 1).
        With 7 groups (race), SVD gives up to 6 directions.
        """
        dim = 512
        embeddings, labels, _ = make_biased_embeddings(dim=dim)
        discoverer = BiasSubspaceDiscoverer(n_bias_directions=4)
        discoverer.fit_image(embeddings, labels)

        B = discoverer.image_bias_directions
        # First dim must match embedding dim
        assert B.shape[0] == dim, f"Expected first dim={dim}, got {B.shape[0]}"
        # Second dim must be at least 1
        assert B.shape[1] >= 1, f"Must have at least 1 bias direction"

    def test_bias_directions_are_unit_vectors(self):
        """Each bias direction should be a unit vector (norm ≈ 1)."""
        embeddings, labels, _ = make_biased_embeddings()
        discoverer = BiasSubspaceDiscoverer(n_bias_directions=3)
        discoverer.fit_image(embeddings, labels)

        B = discoverer.image_bias_directions  # [D, k]
        norms = B.norm(dim=0)  # norm of each column
        assert torch.allclose(norms, torch.ones(3), atol=0.01), (
            f"Bias directions should be unit vectors. Got norms: {norms}"
        )

    def test_detects_known_bias_direction(self):
        """
        The top bias direction should be aligned with the true bias direction
        that we injected into the synthetic data.
        """
        embeddings, labels, true_bias_dir = make_biased_embeddings(
            bias_strength=0.8
        )
        discoverer = BiasSubspaceDiscoverer(n_bias_directions=1)
        discoverer.fit_image(embeddings, labels)

        # The top bias direction should point roughly toward true_bias_dir
        found_direction = discoverer.image_bias_directions[:, 0]

        # Cosine similarity between found and true direction
        # Should be high (close to 1.0) if we detected the right direction
        cosine_sim = abs(
            found_direction.dot(true_bias_dir.float())
            / (found_direction.norm() * true_bias_dir.float().norm())
        ).item()

        assert cosine_sim > 0.7, (
            f"Top bias direction should align with injected bias. "
            f"Got cosine similarity: {cosine_sim:.4f} (expected > 0.7)"
        )

    def test_remove_bias_reduces_group_separation(self):
        """
        After removing bias, the distance between group centroids should
        decrease (the bias is reduced).
        """
        embeddings, labels, _ = make_biased_embeddings(bias_strength=0.5)
        discoverer = BiasSubspaceDiscoverer(n_bias_directions=2)
        discoverer.fit_image(embeddings, labels)

        # Compute group separation BEFORE debiasing
        group0_before = embeddings[labels == 0].mean(dim=0)
        group1_before = embeddings[labels == 1].mean(dim=0)
        dist_before = (group0_before - group1_before).norm().item()

        # Remove bias
        debiased = discoverer.remove_bias_image(embeddings)

        # Compute group separation AFTER debiasing
        group0_after = debiased[labels == 0].mean(dim=0)
        group1_after = debiased[labels == 1].mean(dim=0)
        dist_after = (group0_after - group1_after).norm().item()

        assert dist_after < dist_before, (
            f"Bias removal should reduce group separation. "
            f"Before: {dist_before:.4f}, After: {dist_after:.4f}"
        )

    def test_debiased_embeddings_are_normalized(self):
        """After bias removal, embeddings should still be unit vectors."""
        embeddings, labels, _ = make_biased_embeddings()
        discoverer = BiasSubspaceDiscoverer(n_bias_directions=2)
        discoverer.fit_image(embeddings, labels)
        debiased = discoverer.remove_bias_image(embeddings)

        norms = debiased.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(len(norms)), atol=0.01), (
            f"Debiased embeddings should be unit vectors. "
            f"Max norm deviation: {(norms - 1).abs().max():.6f}"
        )

    def test_remove_bias_before_fit_raises(self):
        """Calling remove_bias before fit should raise a clear error."""
        discoverer = BiasSubspaceDiscoverer()
        embeddings = torch.randn(10, 512)
        with pytest.raises(RuntimeError):
            discoverer.remove_bias_image(embeddings)

    def test_handles_missing_labels(self):
        """Labels of -1 (missing/ambiguous) should be silently ignored."""
        embeddings, labels, _ = make_biased_embeddings(n_per_group=30)
        # Set some labels to -1 (the missing sentinel)
        labels[:10] = -1
        discoverer = BiasSubspaceDiscoverer(n_bias_directions=2)
        # Should not crash
        discoverer.fit_image(embeddings, labels)
        assert discoverer.image_bias_directions is not None

    def test_bias_score_computed(self):
        """get_bias_score should return a non-negative float."""
        embeddings, labels, _ = make_biased_embeddings()
        discoverer = BiasSubspaceDiscoverer(n_bias_directions=2)
        discoverer.fit_image(embeddings, labels)
        score = discoverer.get_bias_score(embeddings, labels, modality="image")
        assert isinstance(score, float)
        assert score >= 0.0


# =============================================================================
# Tests: ProcrustesAligner
# =============================================================================

class TestProcrustesAligner:

    def test_fit_runs_without_error(self):
        """Basic smoke test — fit should not crash."""
        ref = torch.randn(512, 5)
        ref = ref / ref.norm(dim=0, keepdim=True)
        src = torch.randn(512, 5)
        src = src / src.norm(dim=0, keepdim=True)

        aligner = ProcrustesAligner()
        R = aligner.fit(ref, src)
        assert R is not None

    def test_rotation_matrix_shape(self):
        """Rotation matrix should be [k, k]."""
        k = 5
        ref = torch.randn(512, k)
        src = torch.randn(512, k)

        aligner = ProcrustesAligner()
        R = aligner.fit(ref, src)
        assert R.shape == (k, k), f"Expected ({k},{k}), got {R.shape}"

    def test_rotation_is_orthogonal(self):
        """
        Rotation matrix R should satisfy R^T R = I (orthogonality).
        This is what makes it a pure rotation — no stretching or shearing.
        """
        ref = torch.randn(512, 4)
        src = torch.randn(512, 4)

        aligner = ProcrustesAligner()
        R = aligner.fit(ref, src)

        # R^T R should equal identity matrix
        RtR = R.T @ R
        I = torch.eye(4)
        assert torch.allclose(RtR, I, atol=1e-5), (
            f"Rotation matrix is not orthogonal. "
            f"Max deviation from identity: {(RtR - I).abs().max():.6f}"
        )

    def test_rotation_determinant_is_plus_one(self):
        """
        det(R) should be +1 for a pure rotation (not -1 which is a reflection).
        """
        ref = torch.randn(512, 4)
        src = torch.randn(512, 4)

        aligner = ProcrustesAligner()
        R = aligner.fit(ref, src)

        det = torch.det(R).item()
        assert abs(det - 1.0) < 0.01, (
            f"det(R) should be +1 for pure rotation. Got {det:.4f}"
        )

    def test_perfect_alignment_case(self):
        """
        If source IS reference, alignment error should be essentially zero.
        (A rotation of 0 degrees should result in identity matrix)
        """
        ref = torch.randn(512, 3)
        ref = ref / ref.norm(dim=0, keepdim=True)

        aligner = ProcrustesAligner()
        aligner.fit(ref, ref)  # source = reference

        # Alignment error should be close to 0
        assert aligner.alignment_error < 0.01, (
            f"Alignment error for identical inputs should be ~0. "
            f"Got: {aligner.alignment_error:.6f}"
        )

    def test_transform_reduces_alignment_error(self):
        """
        After applying the rotation, aligned source should be closer to
        reference than unaligned source was.
        """
        ref = torch.randn(512, 4)
        ref = ref / ref.norm(dim=0, keepdim=True)
        src = torch.randn(512, 4)
        src = src / src.norm(dim=0, keepdim=True)

        # Distance before alignment
        dist_before = (ref - src).norm(dim=0).mean().item()

        aligner = ProcrustesAligner()
        aligned_src = aligner.fit_transform(ref, src)

        # Distance after alignment
        dist_after = (ref.float() - aligned_src).norm(dim=0).mean().item()

        assert dist_after <= dist_before + 1e-5, (
            f"Alignment should reduce or maintain distance. "
            f"Before: {dist_before:.4f}, After: {dist_after:.4f}"
        )

    def test_transform_before_fit_raises(self):
        """Calling transform before fit should raise RuntimeError."""
        aligner = ProcrustesAligner()
        src = torch.randn(512, 3)
        with pytest.raises(RuntimeError):
            aligner.transform(src)

    def test_mismatched_shapes_raises(self):
        """Reference and source must have same shape."""
        ref = torch.randn(512, 5)
        src = torch.randn(512, 4)  # different k
        aligner = ProcrustesAligner()
        with pytest.raises(ValueError):
            aligner.fit(ref, src)

    def test_fit_transform_same_as_fit_then_transform(self):
        """fit_transform should give same result as fit() then transform()."""
        ref = torch.randn(256, 3)
        src = torch.randn(256, 3)

        # Method 1: fit_transform
        aligner1 = ProcrustesAligner()
        result1 = aligner1.fit_transform(ref, src)

        # Method 2: fit then transform
        aligner2 = ProcrustesAligner()
        aligner2.fit(ref, src)
        result2 = aligner2.transform(src)

        assert torch.allclose(result1, result2, atol=1e-5), (
            "fit_transform and fit+transform should give identical results"
        )


# =============================================================================
# Integration test: Steps IV + V together
# =============================================================================

class TestStepsIVandVIntegration:
    """
    Test that Steps IV (bias subspace) and V (Procrustes) work together.
    This mirrors how they'll be used in the actual training pipeline.
    """

    def test_full_pipeline_iv_to_v(self):
        """
        Simulate the full Steps IV → V pipeline:
        1. Extract biased image + text embeddings (simulated)
        2. Discover bias subspace for each modality (Step IV)
        3. Align text bias to image bias (Step V)
        4. Verify alignment improved
        """
        dim = 256
        n_per_group = 40

        # Simulate image embeddings with gender bias
        image_embs, labels, _ = make_biased_embeddings(
            n_per_group=n_per_group, dim=dim, bias_strength=0.3
        )
        # Simulate text embeddings with gender bias (in DIFFERENT direction)
        text_embs, _, _ = make_biased_embeddings(
            n_per_group=n_per_group, dim=dim, bias_strength=0.3, seed=99
        )

        # Step IV: Discover bias subspaces
        discoverer = BiasSubspaceDiscoverer(n_bias_directions=3)
        discoverer.fit_image(image_embs, labels)
        discoverer.fit_text(text_embs, labels)

        img_dirs = discoverer.image_bias_directions
        txt_dirs = discoverer.text_bias_directions

        # Step V: Align text bias to image bias
        aligner = ProcrustesAligner()
        aligned_txt_dirs = aligner.fit_transform(img_dirs, txt_dirs)

        # After alignment, text directions should be closer to image directions
        dist_before = (img_dirs.float() - txt_dirs.float()).norm(dim=0).mean().item()
        dist_after = (img_dirs.float() - aligned_txt_dirs).norm(dim=0).mean().item()

        assert dist_after <= dist_before + 1e-5, (
            f"Alignment should bring text directions closer to image directions. "
            f"Before: {dist_before:.4f}, After: {dist_after:.4f}"
        )
