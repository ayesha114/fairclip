"""
=============================================================================
Unit tests for the data pipeline
=============================================================================
These tests use SYNTHETIC data so they pass without you having to download
anything. They verify:
    - DemographicBalancedSampler produces correctly balanced batches
    - The Dataset classes return the expected dict shape
    - collate_dict handles mixed types (tensors + strings)

Run with:
    pytest tests/test_dataloader.py -v
=============================================================================
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from data.datasets import (
    FaceDataset, CaptionDataset, collate_dict, make_clip_transform,
)
from data.balanced_sampler import DemographicBalancedSampler


# =============================================================================
# Helpers — build synthetic datasets in a temp directory
# =============================================================================

def make_synthetic_face_manifest(tmp_path: Path, n_per_group: dict[int, int] | None = None) -> Path:
    """
    Create a tiny synthetic FairFace-style manifest with real (random) JPGs.

    Args:
        tmp_path: temp directory
        n_per_group: dict mapping gender_idx -> count. Default: {0: 8, 1: 12}
    """
    if n_per_group is None:
        n_per_group = {0: 8, 1: 12}

    image_dir = tmp_path / "images"
    image_dir.mkdir(exist_ok=True)

    rows = []
    img_idx = 0
    for gender, count in n_per_group.items():
        for _ in range(count):
            # Generate a random RGB image and save it
            arr = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
            path = image_dir / f"img_{img_idx}.jpg"
            Image.fromarray(arr).save(path)

            rows.append({
                "image_path": str(path),
                "split": "train" if img_idx % 5 != 0 else "val",
                "age_idx": img_idx % 9,        # vary across age buckets
                "gender_idx": gender,
                "race_idx": img_idx % 7,       # vary across races
            })
            img_idx += 1

    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def make_synthetic_caption_manifest(tmp_path: Path, n: int = 10) -> Path:
    """Build a synthetic caption manifest."""
    image_dir = tmp_path / "imgs"
    image_dir.mkdir(exist_ok=True)
    rows = []
    for i in range(n):
        arr = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        path = image_dir / f"caption_img_{i}.jpg"
        Image.fromarray(arr).save(path)
        rows.append({
            "image_path": str(path),
            "image_id": i,
            "caption": f"a synthetic caption number {i}",
            "caption_id": i,
            "split": "val2017",
        })
    manifest = tmp_path / "captions.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


# =============================================================================
# Tests: FaceDataset
# =============================================================================

class TestFaceDataset:
    def test_loads_manifest(self, tmp_path):
        manifest = make_synthetic_face_manifest(tmp_path)
        ds = FaceDataset(manifest)
        assert len(ds) == 20  # 8 + 12

    def test_split_filter(self, tmp_path):
        manifest = make_synthetic_face_manifest(tmp_path)
        ds_all = FaceDataset(manifest)
        ds_train = FaceDataset(manifest, split="train")
        ds_val = FaceDataset(manifest, split="val")
        assert len(ds_all) == len(ds_train) + len(ds_val)

    def test_returns_dict_with_correct_keys(self, tmp_path):
        manifest = make_synthetic_face_manifest(tmp_path)
        ds = FaceDataset(manifest)
        sample = ds[0]
        for key in ["image", "age", "gender", "race", "image_path", "index"]:
            assert key in sample, f"Missing key: {key}"

    def test_image_tensor_shape(self, tmp_path):
        manifest = make_synthetic_face_manifest(tmp_path)
        ds = FaceDataset(manifest, image_size=224)
        sample = ds[0]
        assert sample["image"].shape == (3, 224, 224)
        assert sample["image"].dtype == torch.float32

    def test_missing_manifest_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            FaceDataset(tmp_path / "does_not_exist.csv")


# =============================================================================
# Tests: DemographicBalancedSampler
# =============================================================================

class TestBalancedSampler:
    def test_basic_balance(self):
        # 100 samples, 60 male (0), 40 female (1). Sampler should produce
        # batches with equal counts of each.
        labels = np.array([0] * 60 + [1] * 40)
        sampler = DemographicBalancedSampler(labels, batch_size=32, seed=0)

        for batch in sampler.batches():
            counts = np.bincount(labels[batch], minlength=2)
            # Each group should contribute batch_size // 2 = 16 samples
            assert counts[0] == 16
            assert counts[1] == 16

    def test_three_groups(self):
        labels = np.array([0]*30 + [1]*30 + [2]*30)
        sampler = DemographicBalancedSampler(labels, batch_size=30, seed=0)
        for batch in sampler.batches():
            counts = np.bincount(labels[batch], minlength=3)
            # 30 / 3 = 10 each
            assert (counts == 10).all()

    def test_batch_size_not_divisible(self):
        # 33 batch_size, 3 groups -> per_group = 11, effective_batch_size = 33
        # 32 batch_size, 3 groups -> per_group = 10, effective_batch_size = 30
        labels = np.array([0]*50 + [1]*50 + [2]*50)
        sampler = DemographicBalancedSampler(labels, batch_size=32, seed=0)
        assert sampler.effective_batch_size == 30
        assert sampler.per_group == 10

    def test_excludes_minus_one_label(self):
        # Items with label -1 should be silently excluded
        labels = np.array([0]*10 + [1]*10 + [-1]*5)
        sampler = DemographicBalancedSampler(labels, batch_size=4, seed=0)
        # Sampler should ignore the -1s, so num_groups == 2, valid count == 20
        assert sampler.num_groups == 2
        # Check that no batch ever contains a -1 index
        for batch in sampler.batches():
            for idx in batch:
                assert labels[idx] != -1

    def test_batch_size_too_small_raises(self):
        # 3 groups, batch_size 2 -> impossible (need at least one per group)
        labels = np.array([0, 1, 2] * 10)
        with pytest.raises(ValueError):
            DemographicBalancedSampler(labels, batch_size=2)

    def test_set_epoch_changes_order(self):
        # Different epochs should produce different batch orderings
        labels = np.array([0]*20 + [1]*20)
        sampler = DemographicBalancedSampler(labels, batch_size=8, seed=42)
        sampler.set_epoch(0)
        batches_e0 = [list(b) for b in sampler.batches()]
        sampler.set_epoch(1)
        batches_e1 = [list(b) for b in sampler.batches()]
        # At least one batch should differ
        assert batches_e0 != batches_e1

    def test_reproducible_within_epoch(self):
        labels = np.array([0]*20 + [1]*20)
        sampler = DemographicBalancedSampler(labels, batch_size=8, seed=42)
        sampler.set_epoch(0)
        b1 = [list(b) for b in sampler.batches()]
        sampler.set_epoch(0)
        b2 = [list(b) for b in sampler.batches()]
        assert b1 == b2


# =============================================================================
# Tests: collate_dict
# =============================================================================

class TestCollate:
    def test_stacks_tensors(self):
        batch = [
            {"image": torch.zeros(3, 4, 4), "label": 0, "path": "/a"},
            {"image": torch.ones(3, 4, 4), "label": 1, "path": "/b"},
        ]
        out = collate_dict(batch)
        assert out["image"].shape == (2, 3, 4, 4)
        assert out["label"].tolist() == [0, 1]
        assert out["path"] == ["/a", "/b"]

    def test_empty_batch(self):
        assert collate_dict([]) == {}


# =============================================================================
# Tests: CaptionDataset
# =============================================================================

class TestCaptionDataset:
    def test_loads(self, tmp_path):
        manifest = make_synthetic_caption_manifest(tmp_path, n=5)
        ds = CaptionDataset(manifest)
        assert len(ds) == 5
        sample = ds[0]
        assert "image" in sample and "caption" in sample
        assert sample["image"].shape == (3, 224, 224)
        assert isinstance(sample["caption"], str)
        assert len(sample["caption"]) > 0

    def test_max_samples(self, tmp_path):
        manifest = make_synthetic_caption_manifest(tmp_path, n=20)
        ds = CaptionDataset(manifest, max_samples=5)
        assert len(ds) == 5


# =============================================================================
# Tests: image transforms
# =============================================================================

class TestTransforms:
    def test_train_transform_outputs_correct_shape(self):
        transform = make_clip_transform(image_size=224, train=True)
        img = Image.fromarray((np.random.rand(100, 100, 3) * 255).astype(np.uint8))
        out = transform(img)
        assert out.shape == (3, 224, 224)

    def test_eval_transform_deterministic(self):
        transform = make_clip_transform(image_size=224, train=False)
        img = Image.fromarray((np.random.rand(100, 100, 3) * 255).astype(np.uint8))
        out1 = transform(img)
        out2 = transform(img)
        assert torch.allclose(out1, out2)
