"""
COCO Captions Preparation
Configured for your folder structure:
  coco/train2017/         - training images
  coco/val2017/           - validation images
  coco/annotations_trainval2017/annotations/captions_train2017.json
  coco/annotations_trainval2017/annotations/captions_val2017.json
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


def process_split(root, split):
    """Process one COCO split (train2017 or val2017)."""

    # Your annotations are here
    annotations_file = root / "annotations_trainval2017" / "annotations" / f"captions_{split}.json"
    image_dir = root / split  # train2017 or val2017

    if not annotations_file.exists():
        raise FileNotFoundError(f"Annotations not found: {annotations_file}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {image_dir}")

    log.info(f"Loading {annotations_file.name}...")
    with open(annotations_file, "r") as f:
        data = json.load(f)

    # Build image_id -> filename lookup
    id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
    log.info(f"  {len(data['images'])} images, {len(data['annotations'])} captions")

    rows = []
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in id_to_filename:
            continue
        filename = id_to_filename[image_id]
        image_path = image_dir / filename
        rows.append({
            "image_path": str(image_path.resolve()),
            "image_id": image_id,
            "caption_id": ann["id"],
            "caption": ann["caption"].strip(),
            "split": split,
        })

    df = pd.DataFrame(rows)

    # Sample-check images exist
    sample = df.sample(min(20, len(df)), random_state=42)
    missing = [p for p in sample["image_path"] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} sample images not found.\n"
            f"Example: {missing[0]}"
        )

    log.info(f"  Built {len(df)} (image, caption) pairs — verified OK")
    return df


def main(config_path="configs/datasets/local_paths.yaml"):
    log.info("COCO Captions preparation")
    cfg = OmegaConf.load(config_path)
    root = Path(cfg.coco.root)

    if not root.exists():
        raise FileNotFoundError(f"COCO root not found: {root}")

    # Process both splits
    train_df = process_split(root, "train2017")
    val_df = process_split(root, "val2017")

    # Save manifests
    train_path = Path(cfg.coco.manifest_train)
    val_path = Path(cfg.coco.manifest_val)
    train_path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    log.info(f"\nSaved train manifest: {train_path} ({len(train_df)} rows)")
    log.info(f"Saved val manifest:   {val_path} ({len(val_df)} rows)")
    log.info("\nCOCO preparation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets/local_paths.yaml")
    args = parser.parse_args()
    try:
        main(args.config)
    except (FileNotFoundError, ValueError) as e:
        log.error(str(e))
        sys.exit(1)
