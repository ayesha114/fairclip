"""
=============================================================================
Data Verification Script
=============================================================================
One-command sanity check for the entire data pipeline.

Run this AFTER editing configs/datasets/local_paths.yaml and running the
prepare_*.py scripts. It will:
    1. Load each manifest CSV
    2. Build a Dataset for each
    3. Pull a few samples to verify image loading works
    4. Test the demographic-balanced sampler
    5. Print a summary table

If anything fails, the error messages tell you exactly which step broke.

Usage:
    python -m data.verify_data
=============================================================================
"""

import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from data.datasets import (
    FaceDataset, CaptionDataset, FACETDataset, ImageNetValDataset,
    collate_dict,
)
from data.balanced_sampler import DemographicBalancedSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("verify_data")


def check_fairface(cfg) -> bool:
    """Verify FairFace pipeline."""
    log.info("\n=== FairFace ===")
    try:
        ds_train = FaceDataset(cfg.fairface.manifest, split="train", train=True)
        ds_val = FaceDataset(cfg.fairface.manifest, split="val", train=False)
        log.info(f"Train: {len(ds_train)} samples")
        log.info(f"Val:   {len(ds_val)} samples")

        # Pull one sample to verify image loading and transform
        sample = ds_train[0]
        log.info(
            f"Sample: image shape={tuple(sample['image'].shape)}, "
            f"age={sample['age']}, gender={sample['gender']}, race={sample['race']}"
        )

        # Test the balanced sampler with gender as the protected attribute
        labels = ds_train.df["gender_idx"].values
        sampler = DemographicBalancedSampler(labels, batch_size=32, seed=42)
        loader = DataLoader(
            ds_train,
            batch_sampler=sampler.batches(),
            collate_fn=collate_dict,
            num_workers=0,  # 0 for verification (avoids multiprocessing complications)
        )
        batch = next(iter(loader))
        log.info(
            f"Balanced batch: image batch shape={tuple(batch['image'].shape)}, "
            f"gender counts={torch.bincount(batch['gender']).tolist()}"
        )
        return True
    except Exception as e:
        log.error(f"FairFace verification FAILED: {e}")
        return False


def check_utkface(cfg) -> bool:
    log.info("\n=== UTKFace ===")
    try:
        ds = FaceDataset(cfg.utkface.manifest, race_column="race_idx_7")
        log.info(f"Total: {len(ds)} samples")
        sample = ds[0]
        log.info(
            f"Sample: image shape={tuple(sample['image'].shape)}, "
            f"age={sample['age']}, gender={sample['gender']}, race={sample['race']}"
        )
        return True
    except Exception as e:
        log.error(f"UTKFace verification FAILED: {e}")
        return False


def check_coco(cfg) -> bool:
    log.info("\n=== COCO Captions ===")
    try:
        ds_val = CaptionDataset(cfg.coco.manifest_val, max_samples=100)
        log.info(f"Val (sampled to 100): {len(ds_val)} samples")
        sample = ds_val[0]
        log.info(
            f"Sample: image shape={tuple(sample['image'].shape)}, "
            f"caption='{sample['caption'][:60]}...'"
        )
        return True
    except Exception as e:
        log.error(f"COCO verification FAILED: {e}")
        return False


def check_facet(cfg) -> bool:
    log.info("\n=== FACET ===")
    try:
        ds = FACETDataset(cfg.facet.manifest)
        log.info(f"Total: {len(ds)} samples")
        sample = ds[0]
        log.info(
            f"Sample: image shape={tuple(sample['image'].shape)}, "
            f"age={sample['age']}, gender={sample['gender']}"
        )
        return True
    except Exception as e:
        log.error(f"FACET verification FAILED: {e}")
        return False


def check_imagenet(cfg) -> bool:
    if not cfg.imagenet.get("enabled", False):
        log.info("\n=== ImageNet === (disabled in config; skipping)")
        return True
    log.info("\n=== ImageNet ===")
    try:
        ds = ImageNetValDataset(cfg.imagenet.manifest)
        log.info(f"Val: {len(ds)} samples")
        return True
    except Exception as e:
        log.error(f"ImageNet verification FAILED: {e}")
        return False


def main(config_path: str = "configs/datasets/local_paths.yaml") -> None:
    cfg = OmegaConf.load(config_path)

    log.info("=" * 60)
    log.info("FairCLIP — Data Pipeline Verification")
    log.info("=" * 60)

    results = {
        "FairFace": check_fairface(cfg),
        "UTKFace": check_utkface(cfg),
        "COCO": check_coco(cfg),
        "FACET": check_facet(cfg),
        "ImageNet": check_imagenet(cfg),
    }

    log.info("\n" + "=" * 60)
    log.info("Summary")
    log.info("=" * 60)
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        log.info(f"  {name:12s} : {status}")

    if not all(results.values()):
        log.error("\nOne or more datasets failed verification. Fix above errors.")
        sys.exit(1)

    log.info("\nAll datasets verified successfully.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets/local_paths.yaml")
    args = parser.parse_args()
    main(args.config)
