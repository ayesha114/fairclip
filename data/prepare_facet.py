"""
FACET Dataset Preparation
Configured for your folder structure:
  facet/annotations.csv
  facet/imgs_1/*.jpg
  facet/imgs_2/*.jpg
  facet/imgs_3/*.jpg
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
log = logging.getLogger("prepare_facet")


def find_image(filename, image_dirs):
    """Search for image across all image directories."""
    for d in image_dirs:
        path = d / filename
        if path.exists():
            return str(path.resolve())
    return ""


def map_gender(row):
    """Map FACET gender columns to binary 0=Male, 1=Female, -1=ambiguous."""
    masc = float(row.get("gender_presentation_masc", 0) or 0)
    fem = float(row.get("gender_presentation_fem", 0) or 0)
    if masc > fem:
        return 0
    elif fem > masc:
        return 1
    return -1


def map_age(row):
    """Map FACET age columns to approximate bucket index."""
    young  = row.get("age_presentation_young", 0) or 0
    middle = row.get("age_presentation_middle", 0) or 0
    older  = row.get("age_presentation_older", 0) or 0
    if young:
        return 3   # 20-29 bucket
    elif middle:
        return 5   # 40-49 bucket
    elif older:
        return 7   # 60-69 bucket
    return -1


def main(config_path="configs/datasets/local_paths.yaml"):
    log.info("FACET dataset preparation")
    cfg = OmegaConf.load(config_path)
    root = Path(cfg.facet.root)

    if not root.exists():
        raise FileNotFoundError(f"FACET root not found: {root}")

    # Load annotations
    annotations_csv = root / "annotations.csv"
    if not annotations_csv.exists():
        raise FileNotFoundError(f"annotations.csv not found in {root}")

    df = pd.read_csv(annotations_csv)
    log.info(f"Loaded {len(df)} rows")

    # Image directories
    image_dirs = [root / "imgs_1", root / "imgs_2", root / "imgs_3"]
    image_dirs = [d for d in image_dirs if d.exists()]
    log.info(f"Image folders: {[d.name for d in image_dirs]}")

    # Count images per folder
    for d in image_dirs:
        count = len(list(d.glob("*.jpg")))
        log.info(f"  {d.name}: {count} images")

    # Find image column — try common names
    id_col = None
    for candidate in ["filename", "image_filename", "image_id", "person_id"]:
        if candidate in df.columns:
            id_col = candidate
            break
    if id_col is None:
        raise ValueError(f"Cannot find image column. Columns: {list(df.columns)[:10]}")
    log.info(f"Using '{id_col}' as image identifier")

    # Build image paths by searching all 3 folders
    log.info("Building image paths (searching imgs_1, imgs_2, imgs_3)...")
    df["image_path"] = df[id_col].astype(str).apply(
        lambda f: find_image(f if f.endswith(".jpg") else f + ".jpg", image_dirs)
    )

    # Drop rows where image not found
    n_before = len(df)
    df = df[df["image_path"] != ""].copy()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        log.warning(f"Dropped {n_dropped} rows where image not found")
    log.info(f"Matched {len(df)} images")

    # Map gender and age
    df["gender_idx"] = df.apply(map_gender, axis=1)
    df["age_idx"] = df.apply(map_age, axis=1)
    df["race_idx"] = -1  # FACET uses skin tone not race
    df["split"] = "test"  # FACET is evaluation only

    # Save manifest
    out_cols = ["image_path", "split", "age_idx", "gender_idx", "race_idx"]
    out_df = df[out_cols]

    manifest_path = Path(cfg.facet.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(manifest_path, index=False)
    log.info(f"\nSaved manifest: {manifest_path} ({len(out_df)} rows)")

    # Balance report
    log.info("\nGender distribution:")
    log.info(f"  Male      : {(out_df['gender_idx']==0).sum()}")
    log.info(f"  Female    : {(out_df['gender_idx']==1).sum()}")
    log.info(f"  Ambiguous : {(out_df['gender_idx']==-1).sum()}")

    log.info("\nFACET preparation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets/local_paths.yaml")
    args = parser.parse_args()
    try:
        main(args.config)
    except (FileNotFoundError, ValueError) as e:
        log.error(str(e))
        sys.exit(1)
