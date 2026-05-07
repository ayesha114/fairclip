"""
=============================================================================
UTKFace Dataset Preparation
=============================================================================
Builds a manifest CSV for the UTKFace dataset.

UTKFace serves two purposes in our pipeline:
    1. Cross-dataset generalization test (train on FairFace, test on UTKFace)
    2. Secondary fairness training set when comparing against Zhang et al.

Source: Zhang, Song, Qi. "Age Progression/Regression by Conditional GANs"
        CVPR 2017
Download: https://susanqq.github.io/UTKFace/

UTKFace filename format:
    [age]_[gender]_[race]_[date&time].jpg
    e.g.  25_0_0_20170116174525125.jpg.chip.jpg
        - age: 0-116
        - gender: 0=Male, 1=Female
        - race: 0=White, 1=Black, 2=Asian, 3=Indian, 4=Others

NOTE on race mapping: UTKFace's 5-way race split is COARSER than FairFace's
7-way split. We provide TWO race columns:
    race_idx_5  : original UTKFace 5-class encoding
    race_idx_7  : mapped to FairFace's 7 classes (using best-fit approximation)
                  Used for cross-dataset evaluation.

We also bucket UTKFace ages into FairFace's 9 age buckets so the two datasets
can be compared directly.

Usage:
    python -m data.prepare_utkface
=============================================================================
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


# UTKFace's native 5-way race mapping (don't change this; it's defined by the dataset)
UTK_RACE_LABELS_5 = ["White", "Black", "Asian", "Indian", "Others"]

# Best-fit mapping from UTK 5-way to FairFace 7-way.
# "Asian" is split arbitrarily into "East Asian"; "Others" is mapped to
# "Latino_Hispanic" as a rough default. This is an approximation — we report
# the original 5-way labels too so reviewers can see the limitation.
#
# This mirrors how Zhang et al. CVPR 2025 handles cross-dataset evaluation.
UTK_TO_FAIRFACE_RACE = {
    0: 0,  # White -> White
    1: 1,  # Black -> Black
    2: 3,  # Asian -> East Asian (acknowledged simplification)
    3: 5,  # Indian -> Indian
    4: 2,  # Others -> Latino_Hispanic (default)
}

# UTKFace gender is already 0=Male, 1=Female, matching FairFace
UTK_GENDER_LABELS = ["Male", "Female"]

# Bucket continuous age (0-116) into FairFace's 9 buckets
AGE_BUCKETS = [
    (0, 2),    # 0-2
    (3, 9),    # 3-9
    (10, 19),  # 10-19
    (20, 29),  # 20-29
    (30, 39),  # 30-39
    (40, 49),  # 40-49
    (50, 59),  # 50-59
    (60, 69),  # 60-69
    (70, 200),  # more than 70
]


def age_to_bucket(age: int) -> int:
    """Map continuous age to FairFace's 9-bucket index."""
    for idx, (low, high) in enumerate(AGE_BUCKETS):
        if low <= age <= high:
            return idx
    raise ValueError(f"Age {age} not in any bucket")


# UTKFace filename regex.
# Expected pattern: <age>_<gender>_<race>_<datetime>.jpg
# Some files have a trailing ".chip.jpg" — we handle that too.
FILENAME_REGEX = re.compile(r"^(\d+)_(\d)_(\d)_(\d+)\.jpg(\.chip\.jpg)?$")


def parse_filename(filename: str) -> dict | None:
    """
    Parse a UTKFace filename into its components.

    Returns a dict with age/gender/race or None if the filename is malformed.
    Some files in UTKFace are corrupted or have non-standard names — we just
    skip those rather than crash.
    """
    match = FILENAME_REGEX.match(filename)
    if match is None:
        return None

    age = int(match.group(1))
    gender = int(match.group(2))
    race = int(match.group(3))

    # Sanity-check: gender must be 0 or 1, race 0-4
    if gender not in (0, 1):
        return None
    if race not in (0, 1, 2, 3, 4):
        return None
    if age < 0 or age > 200:  # unrealistic ages indicate corruption
        return None

    return {"age": age, "gender": gender, "race_5": race}


def main(config_path: str = "configs/datasets/local_paths.yaml") -> None:
    log.info("UTKFace dataset preparation")

    cfg = OmegaConf.load(config_path)
    root = Path(cfg.utkface.root)

    if not root.exists():
        raise FileNotFoundError(
            f"UTKFace root does not exist: {root}\n"
            f"Edit configs/datasets/local_paths.yaml to set the correct path."
        )

    # UTKFace images can live directly in <root> or in <root>/UTKFace/.
    # We try both.
    image_dir_candidates = [root, root / "UTKFace", root / "utkface", root / "crop_part1"]
    image_dir = None
    for candidate in image_dir_candidates:
        if candidate.exists() and any(candidate.glob("*.jpg")):
            image_dir = candidate
            break
    if image_dir is None:
        raise FileNotFoundError(
            f"No JPG files found in {root} or its expected subdirs. "
            f"Tried: {image_dir_candidates}"
        )
    log.info(f"Image directory: {image_dir}")

    # Collect rows
    rows = []
    skipped = 0
    for img_path in sorted(image_dir.glob("*.jpg")):
        parsed = parse_filename(img_path.name)
        if parsed is None:
            skipped += 1
            continue

        age_idx = age_to_bucket(parsed["age"])
        race_5 = parsed["race_5"]
        race_7 = UTK_TO_FAIRFACE_RACE[race_5]

        rows.append({
            "image_path": str(img_path.resolve()),
            "split": "all",  # UTKFace doesn't have official splits; we'll split later
            "age_raw": parsed["age"],     # original continuous age
            "age_idx": age_idx,            # bucketed to FairFace 9-way
            "gender_idx": parsed["gender"],
            "race_idx_5": race_5,
            "race_idx_7": race_7,
            "gender_str": UTK_GENDER_LABELS[parsed["gender"]],
            "race_str_5": UTK_RACE_LABELS_5[race_5],
        })

    if skipped > 0:
        log.warning(f"Skipped {skipped} files with malformed names")
    log.info(f"Parsed {len(rows)} valid images")

    df = pd.DataFrame(rows)

    manifest_path = Path(cfg.utkface.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)
    log.info(f"Wrote manifest to {manifest_path}")

    # Quick balance report
    log.info("\nGender distribution:")
    for label, count in df["gender_str"].value_counts().items():
        log.info(f"  {label:10s} : {count:6d} ({100*count/len(df):.1f}%)")

    log.info("\nRace (5-way) distribution:")
    for label, count in df["race_str_5"].value_counts().items():
        log.info(f"  {label:10s} : {count:6d} ({100*count/len(df):.1f}%)")

    log.info("\nUTKFace preparation complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets/local_paths.yaml")
    args = parser.parse_args()
    try:
        main(args.config)
    except (FileNotFoundError, ValueError) as e:
        log.error(str(e))
        sys.exit(1)
