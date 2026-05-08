"""
Flickr30k Dataset Preparation
Configured for your folder structure:
  Flickr/flickr30k_images/*.jpg
  Flickr/results.csv  (pipe-separated: image_name| comment_number| comment)
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
log = logging.getLogger("prepare_flickr")


def main(config_path="configs/datasets/local_paths.yaml"):
    log.info("Flickr30k dataset preparation")
    cfg = OmegaConf.load(config_path)
    root = Path(cfg.flickr30k.root)

    if not root.exists():
        raise FileNotFoundError(f"Flickr30k root not found: {root}")

    image_dir = root / "flickr30k_images"
    captions_csv = root / "results.csv"

    if not image_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {image_dir}")
    if not captions_csv.exists():
        raise FileNotFoundError(f"Captions file not found: {captions_csv}")

    # Load captions — pipe separated
    log.info("Loading captions...")
    df = pd.read_csv(captions_csv, sep="|", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    log.info(f"Loaded {len(df)} rows")

    # Build image paths
    df["image_path"] = df["image_name"].astype(str).str.strip().apply(
        lambda f: str((image_dir / f).resolve())
    )
    df["caption"] = df["comment"].astype(str).str.strip()
    df["split"] = "test"

    # Drop missing images
    n_before = len(df)
    df = df[df["image_path"].apply(lambda p: Path(p).exists())].copy()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        log.warning(f"Dropped {n_dropped} rows with missing images")

    log.info(f"Matched {len(df)} (image, caption) pairs")

    # Save manifest
    out_df = df[["image_path", "caption", "split"]].reset_index(drop=True)
    manifest_path = Path(cfg.flickr30k.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(manifest_path, index=False)
    log.info(f"Saved manifest: {manifest_path} ({len(out_df)} rows)")

    # Quick stats
    unique_images = df["image_path"].nunique()
    log.info(f"Unique images: {unique_images}")
    log.info(f"Captions per image: ~{len(df)//unique_images}")
    log.info("\nFlickr30k preparation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets/local_paths.yaml")
    args = parser.parse_args()
    try:
        main(args.config)
    except (FileNotFoundError, ValueError) as e:
        log.error(str(e))
        sys.exit(1)
