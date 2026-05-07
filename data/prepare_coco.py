"""
=============================================================================
COCO Captions Preparation
=============================================================================
Builds manifest CSVs for COCO Captions 2017.

COCO is used for:
    1. Image-text retrieval evaluation (TR@K, IR@K) — proves V-L alignment
       is preserved after debiasing
    2. Joint training pairs for the contrastive loss when not using face data

Source: Lin et al., "Microsoft COCO: Common Objects in Context", ECCV 2014
Captions: Chen et al., "Microsoft COCO Captions", arXiv 2015
Download: https://cocodataset.org/#download

Expected files:
    <root>/train2017/<image_id>.jpg
    <root>/val2017/<image_id>.jpg
    <root>/annotations/captions_train2017.json
    <root>/annotations/captions_val2017.json

Each image has 5 captions. We expand to one row per (image, caption) pair
because that's how CLIP-style training expects pairs.

Usage:
    python -m data.prepare_coco
=============================================================================
"""

import argparse
import json
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
log = logging.getLogger("prepare_coco")


def process_split(root: Path, split: str) -> pd.DataFrame:
    """
    Process one COCO split (train2017 or val2017).

    Returns a DataFrame with one row per (image, caption) pair.
    """
    annotations_file = root / "annotations" / f"captions_{split}.json"
    if not annotations_file.exists():
        raise FileNotFoundError(
            f"COCO annotations not found: {annotations_file}\n"
            f"Did you download the 2017 Train/Val annotations from cocodataset.org?"
        )

    log.info(f"Loading {annotations_file.name}...")
    with open(annotations_file, "r") as f:
        data = json.load(f)

    # COCO JSON structure:
    #   data["images"] = [{"id": int, "file_name": str, "height": int, "width": int}, ...]
    #   data["annotations"] = [{"id": int, "image_id": int, "caption": str}, ...]
    #
    # We build a lookup: image_id -> file_name, then match each caption to an image.

    image_dir = root / split  # e.g., <root>/train2017/
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory missing: {image_dir}")

    log.info(f"Building image lookup ({len(data['images'])} images)...")
    id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

    log.info(f"Processing {len(data['annotations'])} captions...")
    rows = []
    missing_images = 0
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in id_to_filename:
            continue
        filename = id_to_filename[image_id]
        image_path = image_dir / filename

        # Don't verify every file exists — too slow. We sample-check below.
        rows.append({
            "image_path": str(image_path.resolve()),
            "image_id": image_id,
            "caption_id": ann["id"],
            "caption": ann["caption"].strip(),
            "split": split,
        })

    df = pd.DataFrame(rows)

    # Sample-check that images actually exist on disk (catch wrong root path)
    sample = df.sample(min(50, len(df)), random_state=42)
    missing = [p for p in sample["image_path"] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)}/{len(sample)} sampled images not found on disk. "
            f"Example: {missing[0]}\n"
            f"Check the COCO root path in configs/datasets/local_paths.yaml"
        )

    log.info(f"Built {len(df)} (image, caption) pairs for {split}")
    return df


def main(config_path: str = "configs/datasets/local_paths.yaml") -> None:
    log.info("COCO Captions preparation")

    cfg = OmegaConf.load(config_path)
    root = Path(cfg.coco.root)

    if not root.exists():
        raise FileNotFoundError(
            f"COCO root does not exist: {root}\n"
            f"Edit configs/datasets/local_paths.yaml"
        )

    # Process both splits
    train_df = process_split(root, "train2017")
    val_df = process_split(root, "val2017")

    # Write separate manifests because train and val have very different sizes
    # (~590k train pairs vs ~25k val pairs) and we usually load them separately
    train_path = Path(cfg.coco.manifest_train)
    val_path = Path(cfg.coco.manifest_val)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    log.info(f"Wrote train manifest: {train_path} ({len(train_df)} rows)")
    log.info(f"Wrote val manifest:   {val_path} ({len(val_df)} rows)")

    log.info("\nCOCO preparation complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets/local_paths.yaml")
    args = parser.parse_args()
    try:
        main(args.config)
    except (FileNotFoundError, ValueError) as e:
        log.error(str(e))
        sys.exit(1)
