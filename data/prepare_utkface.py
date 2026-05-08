"""
UTKFace Dataset Preparation
Configured for your folder structure:
  utkface/part1/*.jpg
  utkface/part2/*.jpg
  utkface/part3/*.jpg
"""

import argparse
import logging
from pathlib import Path
import re
import sys

import pandas as pd
from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prepare_utkface")

UTK_RACE_LABELS_5 = ["White", "Black", "Asian", "Indian", "Others"]
UTK_GENDER_LABELS = ["Male", "Female"]

UTK_TO_FAIRFACE_RACE = {
    0: 0,  # White -> White
    1: 1,  # Black -> Black
    2: 3,  # Asian -> East Asian
    3: 5,  # Indian -> Indian
    4: 2,  # Others -> Latino_Hispanic
}

AGE_BUCKETS = [
    (0, 2), (3, 9), (10, 19), (20, 29), (30, 39),
    (40, 49), (50, 59), (60, 69), (70, 200),
]

def age_to_bucket(age):
    for idx, (low, high) in enumerate(AGE_BUCKETS):
        if low <= age <= high:
            return idx
    return 8  # default to oldest bucket

FILENAME_REGEX = re.compile(r"^(\d+)_(\d)_(\d)_.*\.jpg$")

def parse_filename(filename):
    match = FILENAME_REGEX.match(filename)
    if match is None:
        return None
    age = int(match.group(1))
    gender = int(match.group(2))
    race = int(match.group(3))
    if gender not in (0, 1):
        return None
    if race not in (0, 1, 2, 3, 4):
        return None
    if age < 0 or age > 200:
        return None
    return {"age": age, "gender": gender, "race_5": race}

def main(config_path="configs/datasets/local_paths.yaml"):
    log.info("UTKFace dataset preparation")
    cfg = OmegaConf.load(config_path)
    root = Path(cfg.utkface.root)

    if not root.exists():
        raise FileNotFoundError(f"UTKFace root not found: {root}")

    # Look in part1, part2, part3
    image_dirs = [root / "part1", root / "part2", root / "part3"]
    found_dirs = [d for d in image_dirs if d.exists()]

    if not found_dirs:
        raise FileNotFoundError(f"No part1/part2/part3 folders found in {root}")

    log.info(f"Found image folders: {[str(d) for d in found_dirs]}")

    rows = []
    skipped = 0

    for image_dir in found_dirs:
        jpg_files = list(image_dir.glob("*.jpg"))
        log.info(f"  {image_dir.name}: {len(jpg_files)} images")

        for img_path in jpg_files:
            parsed = parse_filename(img_path.name)
            if parsed is None:
                skipped += 1
                continue

            age_idx = age_to_bucket(parsed["age"])
            race_5 = parsed["race_5"]
            race_7 = UTK_TO_FAIRFACE_RACE[race_5]

            rows.append({
                "image_path": str(img_path.resolve()),
                "split": "all",
                "age_raw": parsed["age"],
                "age_idx": age_idx,
                "gender_idx": parsed["gender"],
                "race_idx_5": race_5,
                "race_idx_7": race_7,
                "gender_str": UTK_GENDER_LABELS[parsed["gender"]],
                "race_str_5": UTK_RACE_LABELS_5[race_5],
            })

    if skipped > 0:
        log.warning(f"Skipped {skipped} files with malformed names")

    log.info(f"Parsed {len(rows)} valid images total")

    df = pd.DataFrame(rows)

    manifest_path = Path(cfg.utkface.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)
    log.info(f"Saved manifest: {manifest_path}")

    log.info("\nGender distribution:")
    for label, count in df["gender_str"].value_counts().items():
        log.info(f"  {label:10s}: {count:6d} ({100*count/len(df):.1f}%)")

    log.info("\nRace (5-way) distribution:")
    for label, count in df["race_str_5"].value_counts().items():
        log.info(f"  {label:10s}: {count:6d} ({100*count/len(df):.1f}%)")

    log.info("\nUTKFace preparation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets/local_paths.yaml")
    args = parser.parse_args()
    try:
        main(args.config)
    except (FileNotFoundError, ValueError) as e:
        log.error(str(e))
        sys.exit(1)
