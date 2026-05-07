"""
=============================================================================
FairCLIP — PyTorch Dataset Classes
=============================================================================
Dataset classes consumed by the training loop and evaluation scripts.

Design principles:
    1. ALL datasets read from manifest CSVs built by prepare_*.py.
       This decouples dataset format from filesystem layout.
    2. Image transforms come from CLIP's official preprocessing pipeline so
       our embeddings are directly comparable to the pretrained CLIP weights.
    3. Each dataset returns a dict (not a tuple) — this makes it easy to
       extend with new fields without breaking downstream code.

Datasets provided:
    FaceDataset        : FairFace, UTKFace (single image + demographic labels)
    CaptionDataset     : COCO Captions, Flickr30k (image + text caption)
    ClassificationDataset : ImageNet val (image + class label)
    FACETDataset       : FACET (image + demographic labels for evaluation)

All datasets share a common interface so they can be swapped by config.
=============================================================================
"""

import logging
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

log = logging.getLogger(__name__)


# =============================================================================
# Image transforms
# =============================================================================
# We use the same normalization as OpenAI CLIP, which is what every
# CLIP-derived method uses. Hardcoding these constants here (rather than
# importing from clip) makes the code work even if the clip package isn't
# installed (e.g., when only running data preparation).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def make_clip_transform(image_size: int = 224, train: bool = False) -> Callable:
    """
    Build the image transform pipeline for CLIP.

    For training, we add light augmentation (random resized crop, horizontal
    flip). For evaluation, we use deterministic resize + center crop so that
    metrics are reproducible.

    Args:
        image_size: target spatial size (224 for ViT-B/32 and ViT-B/16,
                    336 for ViT-L/14@336)
        train: whether to apply training augmentation

    Returns:
        A callable that takes a PIL image and returns a normalized tensor.
    """
    if train:
        return T.Compose([
            # RandomResizedCrop is the standard for CLIP fine-tuning. The
            # scale range (0.9, 1.0) is conservative because face crops can
            # lose demographic-relevant features under aggressive crops.
            T.RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])
    else:
        # Evaluation transform: deterministic, matches OpenAI CLIP exactly.
        return T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])


# =============================================================================
# Face datasets (FairFace, UTKFace)
# =============================================================================

class FaceDataset(Dataset):
    """
    Dataset for face images with demographic attribute labels.

    Works for both FairFace and UTKFace (since both manifests share the
    columns: image_path, split, age_idx, gender_idx, race_idx).

    Each item returned is a dict:
        {
            "image": Tensor [3, H, W]      — normalized image
            "age":    int (LongTensor scalar) — age bucket index
            "gender": int                       — 0 or 1
            "race":   int                       — 0..6 (or -1 if missing)
            "image_path": str                   — for debugging only
            "index":  int                       — row index in manifest
        }
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str | None = None,
        image_size: int = 224,
        train: bool = False,
        race_column: str = "race_idx",  # UTKFace uses "race_idx_7" for cross-dataset
    ):
        """
        Args:
            manifest_path: CSV file built by prepare_fairface.py / prepare_utkface.py
            split: 'train', 'val', or None to use all rows
            image_size: target image size in pixels
            train: whether to apply training augmentation
            race_column: which race column to read (allows UTK 5-way vs 7-way)
        """
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}\n"
                f"Run the corresponding prepare_*.py script first."
            )

        df = pd.read_csv(manifest_path)
        log.info(f"Loaded manifest {manifest_path.name}: {len(df)} rows")

        # Filter by split if requested
        if split is not None:
            if "split" not in df.columns:
                raise ValueError(f"Manifest has no 'split' column; cannot filter to {split}")
            df = df[df["split"] == split].reset_index(drop=True)
            log.info(f"Filtered to split='{split}': {len(df)} rows")

        # Sanity-check required columns
        required = {"image_path", "age_idx", "gender_idx"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Manifest is missing required columns: {missing}")

        # Race column may be named differently for UTKFace
        if race_column not in df.columns:
            log.warning(f"Race column '{race_column}' not in manifest; race will be -1")
            df[race_column] = -1

        self.df = df
        self.race_column = race_column
        self.transform = make_clip_transform(image_size=image_size, train=train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]

        # Load and transform the image. We catch image errors here rather than
        # silently skipping — if a file is corrupt, we want to know.
        try:
            image = Image.open(row["image_path"]).convert("RGB")
        except (OSError, IOError) as e:
            raise IOError(f"Failed to load image {row['image_path']}: {e}")

        image_tensor = self.transform(image)

        return {
            "image": image_tensor,
            "age": int(row["age_idx"]),
            "gender": int(row["gender_idx"]),
            "race": int(row[self.race_column]),
            "image_path": row["image_path"],
            "index": idx,
        }


# =============================================================================
# Caption datasets (COCO, Flickr30k)
# =============================================================================

class CaptionDataset(Dataset):
    """
    Dataset for (image, caption) pairs.

    Used for retrieval evaluation and as the V-L alignment training signal
    when contrastive learning needs natural image-text pairs.

    Each item returned is a dict:
        {
            "image": Tensor [3, H, W]
            "caption": str             — raw text (NOT tokenized; the model
                                         tokenizer happens in the trainer)
            "image_id": int
            "image_path": str
            "index": int
        }

    Note: We deliberately do NOT tokenize here because different CLIP models
    use different tokenizers (OpenAI clip vs. open_clip). Tokenization is
    deferred to the training loop where the model is known.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        image_size: int = 224,
        train: bool = False,
        max_samples: int | None = None,  # for fast debugging
    ):
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        df = pd.read_csv(manifest_path)
        log.info(f"Loaded {manifest_path.name}: {len(df)} rows")

        if max_samples is not None:
            df = df.head(max_samples).reset_index(drop=True)
            log.info(f"Subsampled to {len(df)} rows (debug mode)")

        required = {"image_path", "caption"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {missing}")

        self.df = df
        self.transform = make_clip_transform(image_size=image_size, train=train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        try:
            image = Image.open(row["image_path"]).convert("RGB")
        except (OSError, IOError) as e:
            raise IOError(f"Failed to load image {row['image_path']}: {e}")

        return {
            "image": self.transform(image),
            "caption": str(row["caption"]),
            "image_id": int(row.get("image_id", idx)),
            "image_path": row["image_path"],
            "index": idx,
        }


# =============================================================================
# FACET dataset
# =============================================================================

class FACETDataset(Dataset):
    """
    Dataset for FACET out-of-domain fairness evaluation.

    Identical interface to FaceDataset but reads FACET-specific labels.
    Rows where gender or age is -1 (ambiguous in FACET) are kept in the
    manifest but the evaluation code filters them per-attribute.
    """

    def __init__(self, manifest_path: str | Path, image_size: int = 224):
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"FACET manifest not found: {manifest_path}")
        self.df = pd.read_csv(manifest_path)
        self.transform = make_clip_transform(image_size=image_size, train=False)
        log.info(f"FACET dataset loaded: {len(self.df)} images")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        try:
            image = Image.open(row["image_path"]).convert("RGB")
        except (OSError, IOError) as e:
            raise IOError(f"Failed to load image {row['image_path']}: {e}")

        return {
            "image": self.transform(image),
            "age": int(row.get("age_idx", -1)),
            "gender": int(row.get("gender_idx", -1)),
            "race": int(row.get("race_idx", -1)),
            "image_path": row["image_path"],
            "index": idx,
        }


# =============================================================================
# ImageNet validation dataset
# =============================================================================

class ImageNetValDataset(Dataset):
    """
    ImageNet validation set for zero-shot classification evaluation.

    This is used to verify that our debiasing does NOT hurt the model's
    general V-L alignment ability (a standard sanity check in the literature).

    Reads a manifest with columns: image_path, class_idx, class_name
    """

    def __init__(self, manifest_path: str | Path, image_size: int = 224):
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"ImageNet manifest not found: {manifest_path}")
        self.df = pd.read_csv(manifest_path)
        self.transform = make_clip_transform(image_size=image_size, train=False)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        return {
            "image": self.transform(image),
            "class_idx": int(row["class_idx"]),
            "class_name": str(row.get("class_name", "")),
            "image_path": row["image_path"],
        }


# =============================================================================
# Utility: collate function
# =============================================================================

def collate_dict(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Custom collate function for our dict-based datasets.

    PyTorch's default collate works on tuples. Our datasets return dicts,
    and some fields (like 'image_path' and 'caption') are strings that can't
    be stacked. This function stacks tensors and gathers strings into lists.
    """
    if len(batch) == 0:
        return {}

    out: dict[str, Any] = {}
    keys = batch[0].keys()
    for key in keys:
        values = [item[key] for item in batch]
        first = values[0]
        if isinstance(first, torch.Tensor):
            out[key] = torch.stack(values, dim=0)
        elif isinstance(first, (int, float, bool)):
            out[key] = torch.tensor(values)
        else:
            # strings, paths, etc. — keep as a list
            out[key] = values
    return out
