"""
FairFace Dataset Preparation
Builds a clean manifest CSV for FairFace dataset.
Configured for your specific folder structure:
  images/train/   - training images
  images/val/     - validation images
  images/labels/train/train_labels.csv
  images/labels/val/val_labels.csv
"""

import argparse
import logging
from pathlib import Path
import sys

import pandas as pd
from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prepare_fairface")

# Label encoders — integers that match FairFace paper ordering
AGE_LABELS = [
    "0-2", "3-9", "10-19", "20-29", "30-39",
    "40-49", "50-59", "60-69", "more than 70",
]
AGE_TO_IDX = {label: idx for idx, label in enumerate(AGE_LABELS)}

GENDER_LABELS = ["Male", "Female"]
GENDER_TO_IDX = {label: idx for idx, label in enumerate(GENDER_LABELS)}

RACE_LABELS = [
    "White", "Black", "Latino_Hispanic", "East Asian",
    "Southeast Asian", "Indian", "Middle Eastern",
]
RACE_TO_IDX = {label: idx for idx, label in enumerate(RACE_LABELS)}


def main(config_path="configs/datasets/local_paths.yaml"):
    log.info("FairFace dataset preparation")
    cfg = OmegaConf.load(config_path)

    root = Path(cfg.fairface.root)
    train_csv = Path(cfg.fairface.train_csv)
    val_csv = Path(cfg.fairface.val_csv)

    # Check paths exist
    for path in [root, train_csv, val_csv]:
        if not path.exists():
            raise FileNotFoundError(f"Not found: {path}")

    all_rows = []

    for csv_path, split in [(train_csv, "train"), (val_csv, "val")]:
        log.info(f"Processing {split} split: {csv_path}")
        df = pd.read_csv(csv_path)
        log.info(f"  Loaded {len(df)} rows")

        # Build absolute image paths
        # CSV 'file' column has values like 'train/1.jpg' or 'val/1.jpg'
        # root is the 'images' folder, so: images/train/1.jpg
        df["image_path"] = df["file"].apply(
            lambda f: str((root / f).resolve())
        )
        df["split"] = split

        # Encode labels as integers
        df["age_idx"] = df["age"].map(AGE_TO_IDX)
        df["gender_idx"] = df["gender"].map(GENDER_TO_IDX)
        df["race_idx"] = df["race"].map(RACE_TO_IDX)

        # Check for any unmapped labels
        for col in ["age_idx", "gender_idx", "race_idx"]:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                bad = df.loc[df[col].isna(), col.replace("_idx","")].unique()
                log.warning(f"  {n_missing} unmapped {col} values: {bad}")

        df["age_idx"] = df["age_idx"].fillna(-1).astype(int)
        df["gender_idx"] = df["gender_idx"].fillna(-1).astype(int)
        df["race_idx"] = df["race_idx"].fillna(-1).astype(int)

        # Verify sample images exist
        sample = df.sample(min(20, len(df)), random_state=42)
        missing = [p for p in sample["image_path"] if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} sample images not found.\n"
                f"Example: {missing[0]}\n"
                f"Check the root path in local_paths.yaml"
            )
        log.info(f"  Image paths verified OK")

        all_rows.append(df)

    # Combine train + val
    combined = pd.concat(all_rows, ignore_index=True)
    combined = combined.rename(columns={
        "age": "age_str", "gender": "gender_str", "race": "race_str"
    })

    output_cols = [
        "image_path", "split",
        "age_idx", "gender_idx", "race_idx",
        "age_str", "gender_str", "race_str",
    ]
    combined = combined[output_cols]

    # Save manifest
    manifest_path = Path(cfg.fairface.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(manifest_path, index=False)
    log.info(f"\nSaved manifest: {manifest_path} ({len(combined)} rows)")

    # Balance report
    log.info("\nGender distribution:")
    for label, count in combined["gender_str"].value_counts().items():
        log.info(f"  {label:10s}: {count:6d} ({100*count/len(combined):.1f}%)")

    log.info("\nRace distribution:")
    for label, count in combined["race_str"].value_counts().items():
        log.info(f"  {label:20s}: {count:6d} ({100*count/len(combined):.1f}%)")

    log.info("\nFairFace preparation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets/local_paths.yaml")
    args = parser.parse_args()
    try:
        main(args.config)
    except (FileNotFoundError, ValueError) as e:
        log.error(str(e))
        sys.exit(1)
