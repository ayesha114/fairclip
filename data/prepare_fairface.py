"""
=============================================================================
FairFace Dataset Preparation
=============================================================================
Builds a clean manifest CSV for the FairFace dataset that our PyTorch
Dataset class will consume.

FairFace is the PRIMARY dataset for our fairness training (Step I of the
proposal methodology). It contains ~108k face images with balanced
demographics across 7 races, 9 age groups, and 2 genders.

Source: Karkkainen & Joo, WACV 2021
Download: https://github.com/joojs/fairface

What this script does:
    1. Reads the official fairface_label_{train,val}.csv files
    2. Constructs absolute image paths (verifies each file exists)
    3. Encodes demographic labels as integers
    4. Writes a single combined manifest with a 'split' column
    5. Reports class balance for each demographic attribute

Output manifest columns:
    image_path  : absolute path to JPG
    split       : 'train' or 'val'
    age         : 0-8 (9 age buckets)
    gender      : 0=Male, 1=Female
    race        : 0-6 (7 race categories)
    age_str, gender_str, race_str : human-readable labels

Usage:
    python -m data.prepare_fairface
=============================================================================
"""

# Standard library imports go first (PEP 8)
import argparse
import logging
from pathlib import Path
import sys

# Third-party imports
import pandas as pd
from omegaconf import OmegaConf

# Set up logging — using the 'logging' module rather than print() so that
# verbosity can be controlled with --quiet / --verbose flags later.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prepare_fairface")


# =============================================================================
# Label encoders
# =============================================================================
# We encode the categorical labels as integers because PyTorch loss functions
# expect integer class indices, not strings. The mappings are kept HERE rather
# than in the dataset class so that anyone reading the manifest CSV can see
# exactly what each integer means.
#
# IMPORTANT: Do NOT change the order. These match the standard ordering used
# in the FairFace paper and in our baseline (Zhang et al. CVPR 2025), so our
# results are directly comparable.

# 9 age buckets (FairFace ordering)
AGE_LABELS = [
    "0-2", "3-9", "10-19", "20-29", "30-39",
    "40-49", "50-59", "60-69", "more than 70",
]
AGE_TO_IDX = {label: idx for idx, label in enumerate(AGE_LABELS)}

# 2 gender categories
GENDER_LABELS = ["Male", "Female"]
GENDER_TO_IDX = {label: idx for idx, label in enumerate(GENDER_LABELS)}

# 7 race categories (FairFace ordering)
RACE_LABELS = [
    "White", "Black", "Latino_Hispanic", "East Asian",
    "Southeast Asian", "Indian", "Middle Eastern",
]
RACE_TO_IDX = {label: idx for idx, label in enumerate(RACE_LABELS)}


def load_fairface_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load and validate one of FairFace's label CSVs.

    The official FairFace CSV has columns:
        file, age, gender, race, service_test

    We don't use 'service_test' (it's an internal flag from the paper authors).
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"FairFace label CSV not found: {csv_path}\n"
            f"Did you download fairface_label_{{train,val}}.csv from "
            f"https://github.com/joojs/fairface and place it in the dataset root?"
        )

    df = pd.read_csv(csv_path)
    log.info(f"Loaded {len(df)} rows from {csv_path.name}")

    # Sanity-check the columns. If FairFace ever changes its CSV schema,
    # this is where we want to fail loudly rather than produce silent garbage.
    required_cols = {"file", "age", "gender", "race"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"FairFace CSV is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    return df


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert string labels to integer indices.

    Adds three new integer columns: age_idx, gender_idx, race_idx.
    Keeps the original string columns for human readability.
    """
    # We use .map() with our explicit label dicts. If a row has an unexpected
    # label (e.g. a typo in FairFace), .map() returns NaN, which we catch.
    df = df.copy()  # never mutate the caller's dataframe
    df["age_idx"] = df["age"].map(AGE_TO_IDX)
    df["gender_idx"] = df["gender"].map(GENDER_TO_IDX)
    df["race_idx"] = df["race"].map(RACE_TO_IDX)

    # Check for unmapped values
    for col in ["age_idx", "gender_idx", "race_idx"]:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            unique_unmapped = df.loc[df[col].isna(), col.replace("_idx", "")].unique()
            raise ValueError(
                f"{n_missing} rows have unmappable {col} values: {unique_unmapped}. "
                f"Check the label encoders at the top of this file."
            )

    # After the check, we can safely cast to int (they're floats now because of NaN handling)
    df["age_idx"] = df["age_idx"].astype(int)
    df["gender_idx"] = df["gender_idx"].astype(int)
    df["race_idx"] = df["race_idx"].astype(int)

    return df


def build_image_paths(df: pd.DataFrame, root: Path, split: str) -> pd.DataFrame:
    """
    Build absolute image paths and verify each file exists.

    FairFace's CSV 'file' column has values like 'train/1.jpg' or 'val/1.jpg',
    so we just join with the dataset root.
    """
    df = df.copy()
    df["image_path"] = df["file"].apply(lambda f: str((root / f).resolve()))
    df["split"] = split

    # Verify a sample of paths exist. We don't check ALL of them because that
    # takes a long time on disk; checking ~100 random paths catches most errors
    # (wrong root, wrong split name, missing files).
    sample = df.sample(min(100, len(df)), random_state=42)
    missing = [p for p in sample["image_path"] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)}/{len(sample)} sampled images do not exist. "
            f"Example missing path: {missing[0]}\n"
            f"Check the 'root' field in configs/datasets/local_paths.yaml"
        )

    return df


def report_balance(df: pd.DataFrame) -> None:
    """
    Print demographic balance for each attribute.

    This is a sanity check — FairFace claims to be balanced, and if our
    counts come out wildly skewed, something went wrong upstream.
    """
    log.info("=" * 60)
    log.info("Demographic balance report:")
    log.info("=" * 60)

    for attr, labels in [
        ("gender", GENDER_LABELS),
        ("race", RACE_LABELS),
        ("age", AGE_LABELS),
    ]:
        counts = df[attr].value_counts().reindex(labels, fill_value=0)
        log.info(f"\n{attr.upper()}:")
        for label, count in counts.items():
            pct = 100 * count / len(df)
            log.info(f"  {label:25s} : {count:6d} ({pct:5.2f}%)")


def main(config_path: str = "configs/datasets/local_paths.yaml") -> None:
    """Main entry point — orchestrates the full preparation pipeline."""
    log.info("FairFace dataset preparation")
    log.info(f"Loading config from: {config_path}")

    cfg = OmegaConf.load(config_path)
    root = Path(cfg.fairface.root)

    # Sanity-check the root exists and is a directory
    if not root.exists():
        raise FileNotFoundError(
            f"FairFace root does not exist: {root}\n"
            f"Edit configs/datasets/local_paths.yaml to set the correct path."
        )
    if not root.is_dir():
        raise NotADirectoryError(f"FairFace root is not a directory: {root}")

    log.info(f"FairFace root: {root}")

    # Load both train and val splits
    train_df = load_fairface_csv(root / "fairface_label_train.csv")
    val_df = load_fairface_csv(root / "fairface_label_val.csv")

    # Encode integer labels and build absolute paths for each split separately
    train_df = encode_labels(train_df)
    train_df = build_image_paths(train_df, root, split="train")

    val_df = encode_labels(val_df)
    val_df = build_image_paths(val_df, root, split="val")

    # Combine into a single manifest. We keep BOTH splits in one CSV with a
    # 'split' column rather than two separate CSVs because:
    # (a) it's easier to compute global statistics (e.g. demographic balance)
    # (b) some experiments need to use the val set for fitting bias subspace
    combined = pd.concat([train_df, val_df], ignore_index=True)

    # Pick the columns we want to keep, in a sensible order
    output_cols = [
        "image_path", "split",
        "age_idx", "gender_idx", "race_idx",
        "age", "gender", "race",  # keep human-readable labels too
    ]
    combined = combined[output_cols]

    # Rename for consistency with our convention: <attr>_idx is the integer,
    # <attr>_str is the human-readable label.
    combined = combined.rename(columns={
        "age": "age_str", "gender": "gender_str", "race": "race_str",
    })

    # Write the manifest
    manifest_path = Path(cfg.fairface.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(manifest_path, index=False)
    log.info(f"\nWrote manifest with {len(combined)} rows to {manifest_path}")

    # Report demographic balance using the human-readable labels
    report_balance(combined.rename(columns={
        "age_str": "age", "gender_str": "gender", "race_str": "race",
    }))

    log.info("\n" + "=" * 60)
    log.info("FairFace preparation complete")
    log.info("=" * 60)


if __name__ == "__main__":
    # We accept an optional --config flag so that testing can pass in a
    # different config file without polluting the production config.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/datasets/local_paths.yaml",
        help="Path to the local_paths.yaml config file",
    )
    args = parser.parse_args()

    try:
        main(args.config)
    except (FileNotFoundError, ValueError, NotADirectoryError) as e:
        # Friendly error reporting — we KNOW these are user-facing errors,
        # so we print a clean message instead of dumping a stack trace.
        log.error(str(e))
        sys.exit(1)
