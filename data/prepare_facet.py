"""
=============================================================================
FACET Dataset Preparation
=============================================================================
Builds a manifest CSV for the FACET dataset.

FACET serves as the OUT-OF-DOMAIN fairness benchmark, exactly matching the
evaluation protocol of Zhang et al. CVPR 2025. Models trained on FairFace
are evaluated on FACET to test generalization.

Source: Gustafson et al., "FACET: Fairness in Computer Vision Evaluation
        Benchmark", ICCV 2023, Meta AI
Download: https://facet.metademolab.com/

FACET attributes used in our pipeline:
    - gender_presentation_masc / gender_presentation_fem (binary perceived gender)
    - age_presentation_young / middle / older (3 age buckets)
    - skin_tone_1 .. skin_tone_10 (Monk Skin Tone scale, 10 levels)

We map FACET's annotations to the same attribute schema as FairFace where
possible (gender, coarse age) so cross-dataset evaluation is straightforward.

Usage:
    python -m data.prepare_facet
=============================================================================
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


def map_facet_gender(row: pd.Series) -> int:
    """
    Map FACET gender presentation columns to a binary label compatible with
    FairFace's encoding (0=Male, 1=Female).

    FACET has separate masc/fem columns because gender presentation is
    perceived rather than self-reported. We use the dominant column.

    Returns -1 if neither column has a strong signal (these rows are
    excluded from gender evaluation).
    """
    masc = row.get("gender_presentation_masc", 0) or 0
    fem = row.get("gender_presentation_fem", 0) or 0
    if masc > fem:
        return 0  # masc -> Male (FairFace 0)
    elif fem > masc:
        return 1  # fem -> Female (FairFace 1)
    else:
        return -1  # ambiguous; will be filtered out for gender eval


def map_facet_age(row: pd.Series) -> int:
    """
    Map FACET 3-way age to FairFace's 9 buckets using approximate centers.

    FACET only has 3 ages; we map to the central FairFace bucket of each:
        young   -> bucket 3 (20-29)
        middle  -> bucket 5 (40-49)
        older   -> bucket 7 (60-69)
    """
    if row.get("age_presentation_young", 0):
        return 3
    elif row.get("age_presentation_middle", 0):
        return 5
    elif row.get("age_presentation_older", 0):
        return 7
    return -1  # no age annotation


def main(config_path: str = "configs/datasets/local_paths.yaml") -> None:
    log.info("FACET dataset preparation")

    cfg = OmegaConf.load(config_path)
    root = Path(cfg.facet.root)

    if not root.exists():
        raise FileNotFoundError(
            f"FACET root does not exist: {root}\n"
            f"Edit configs/datasets/local_paths.yaml"
        )

    # FACET ships an "annotations.csv" or similar. Look for any CSV in the root.
    csv_candidates = list(root.glob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(
            f"No CSV annotation file found in {root}. "
            f"Did you download the FACET annotations?"
        )

    # Prefer the most descriptive filename match
    annotations_csv = None
    for c in csv_candidates:
        if "annotation" in c.name.lower() or "facet" in c.name.lower():
            annotations_csv = c
            break
    if annotations_csv is None:
        annotations_csv = csv_candidates[0]
    log.info(f"Using annotations file: {annotations_csv}")

    df = pd.read_csv(annotations_csv)
    log.info(f"Loaded {len(df)} FACET rows")
    log.info(f"Columns: {list(df.columns)[:10]}...")  # first 10 only

    # Find the image directory. FACET typically uses 'images/' or similar.
    image_dir_candidates = [root / "images", root / "imgs", root]
    image_dir = next((d for d in image_dir_candidates if d.exists() and any(d.glob("*.jpg")) or any(d.glob("*.png"))), None)
    if image_dir is None:
        raise FileNotFoundError(f"Could not find image directory in {root}")
    log.info(f"Image directory: {image_dir}")

    # FACET's image identifier column varies — try common names
    id_col = None
    for candidate in ["filename", "image_filename", "image_id", "image_name", "person_id"]:
        if candidate in df.columns:
            id_col = candidate
            break
    if id_col is None:
        raise ValueError(
            f"Could not find image filename column in FACET CSV. "
            f"Available columns: {list(df.columns)}"
        )
    log.info(f"Using '{id_col}' as image identifier")

    # Build absolute paths
    def resolve_path(name: str) -> str:
        # Some FACET versions store just the ID without extension
        candidate = image_dir / name
        if candidate.exists():
            return str(candidate.resolve())
        # Try common extensions
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate_ext = image_dir / f"{name}{ext}"
            if candidate_ext.exists():
                return str(candidate_ext.resolve())
        return ""  # missing

    df["image_path"] = df[id_col].astype(str).apply(resolve_path)

    # Drop rows with missing images
    n_before = len(df)
    df = df[df["image_path"] != ""].copy()
    if len(df) < n_before:
        log.warning(f"Dropped {n_before - len(df)} rows with missing images")

    # Map gender and age
    df["gender_idx"] = df.apply(map_facet_gender, axis=1)
    df["age_idx"] = df.apply(map_facet_age, axis=1)

    # We don't have a direct race mapping from skin tone, so we leave race_idx
    # empty for FACET. Race evaluation on FACET is optional and noted in the
    # paper as a limitation.
    df["race_idx"] = -1

    # Output schema-compatible with our other manifests
    df["split"] = "test"  # FACET is only used for evaluation

    output_cols = ["image_path", "split", "age_idx", "gender_idx", "race_idx"]
    out_df = df[output_cols]

    manifest_path = Path(cfg.facet.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(manifest_path, index=False)
    log.info(f"Wrote manifest with {len(out_df)} rows to {manifest_path}")

    # Balance reports
    log.info("\nGender distribution (after mapping):")
    log.info(f"  Male       : {(out_df['gender_idx'] == 0).sum()}")
    log.info(f"  Female     : {(out_df['gender_idx'] == 1).sum()}")
    log.info(f"  Ambiguous  : {(out_df['gender_idx'] == -1).sum()}  (excluded from gender eval)")

    log.info("\nAge bucket distribution:")
    for bucket in [3, 5, 7, -1]:
        n = (out_df["age_idx"] == bucket).sum()
        label = "ambiguous" if bucket == -1 else f"bucket {bucket}"
        log.info(f"  {label:12s} : {n}")

    log.info("\nFACET preparation complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets/local_paths.yaml")
    args = parser.parse_args()
    try:
        main(args.config)
    except (FileNotFoundError, ValueError) as e:
        log.error(str(e))
        sys.exit(1)
