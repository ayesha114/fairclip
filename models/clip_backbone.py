"""
=============================================================================
FairCLIP — CLIP Backbone Wrapper (Step III of Methodology)
=============================================================================
This file wraps the pretrained CLIP model so the rest of FairCLIP can use it
cleanly without worrying about which version of CLIP is installed.

What this file does (in plain English):
    - Loads a pretrained CLIP model (ViT-B/32 or ViT-B/16)
    - Provides two functions: encode_images() and encode_text()
    - Returns normalized embeddings ready for bias analysis
    - Supports switching backbones via a simple config string

Why we need this:
    Step III of the proposal says "Extract features using CLIP."
    Every other step (PCA, Procrustes, fairness loss) uses these embeddings.
    This file is the foundation everything else builds on.

Usage:
    backbone = CLIPBackbone(model_name="ViT-B/32", device="cuda")
    image_embeddings = backbone.encode_images(images)   # [N, 512]
    text_embeddings  = backbone.encode_text(texts)      # [N, 512]
=============================================================================
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class CLIPBackbone(nn.Module):
    """
    Wrapper around pretrained CLIP that gives us clean image and text encoders.

    Why a wrapper and not raw CLIP?
    Because raw CLIP has slightly different APIs depending on whether you use
    the OpenAI version or OpenCLIP version. This wrapper hides those differences
    so the rest of the code is clean and consistent.
    """

    # Supported model names and their embedding dimensions
    # These are standard CLIP models — we use ViT-B/32 and ViT-B/16
    SUPPORTED_MODELS = {
        "ViT-B/32": 512,   # faster, less memory, good for RTX 3050
        "ViT-B/16": 512,   # slightly better quality, same embedding size
        "ViT-L/14": 768,   # large model — needs 24GB GPU
    }

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = "cuda",
        freeze_backbone: bool = True,
    ):
        """
        Args:
            model_name: Which CLIP backbone to use. Options:
                        "ViT-B/32" — recommended for RTX 3050 (8GB)
                        "ViT-B/16" — better quality, needs ~10GB
                        "ViT-L/14" — best quality, needs 24GB
            device: "cuda" to use GPU, "cpu" for testing only
            freeze_backbone: If True, CLIP weights are frozen (not trained).
                            We set this to True initially, then selectively
                            unfreeze during fairness-aware contrastive learning.
        """
        super().__init__()

        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Choose from: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.model_name = model_name
        self.device = device
        self.embedding_dim = self.SUPPORTED_MODELS[model_name]

        log.info(f"Loading CLIP backbone: {model_name}")
        log.info(f"Embedding dimension: {self.embedding_dim}")
        log.info(f"Device: {device}")

        # Load pretrained CLIP
        # We try OpenAI CLIP first, fall back to OpenCLIP if not available
        self.clip_model, self.preprocess = self._load_clip(model_name, device)

        # Store the tokenizer (used for text encoding)
        self.tokenizer = self._load_tokenizer()

        # Freeze backbone weights if requested
        # Frozen = CLIP weights don't change during training
        # This preserves CLIP's pretrained knowledge while we add fairness
        if freeze_backbone:
            self._freeze_backbone()
            log.info("Backbone frozen — CLIP weights will not be updated")
        else:
            log.info("Backbone unfrozen — CLIP weights will be updated during training")

        self.to(device)

    def _load_clip(self, model_name: str, device: str):
        """
        Load pretrained CLIP model.
        Tries OpenAI CLIP first, then OpenCLIP as backup.
        """
        try:
            import clip
            model, preprocess = clip.load(model_name, device=device)
            log.info("Loaded OpenAI CLIP successfully")
            self._clip_type = "openai"
            return model, preprocess
        except Exception as e:
            log.warning(f"OpenAI CLIP failed ({e}), trying OpenCLIP...")

        try:
            import open_clip
            # Map OpenAI names to OpenCLIP names
            name_map = {
                "ViT-B/32": ("ViT-B-32", "openai"),
                "ViT-B/16": ("ViT-B-16", "openai"),
                "ViT-L/14": ("ViT-L-14", "openai"),
            }
            arch, pretrained = name_map[model_name]
            model, _, preprocess = open_clip.create_model_and_transforms(
                arch, pretrained=pretrained, device=device
            )
            log.info("Loaded OpenCLIP successfully")
            self._clip_type = "openclip"
            return model, preprocess
        except Exception as e:
            raise RuntimeError(
                f"Could not load CLIP. Tried both OpenAI CLIP and OpenCLIP.\n"
                f"Last error: {e}\n"
                f"Install with: pip install git+https://github.com/openai/CLIP.git"
            )

    def _load_tokenizer(self):
        """Load the text tokenizer matching the loaded CLIP type."""
        if self._clip_type == "openai":
            import clip
            return clip.tokenize
        else:
            import open_clip
            return open_clip.get_tokenizer(
                self.model_name.replace("/", "-").replace("ViT-", "ViT-")
            )

    def _freeze_backbone(self):
        """Freeze all CLIP parameters so they don't update during training."""
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """
        Unfreeze CLIP parameters for full fine-tuning.
        Called during fairness-aware contrastive learning (Step VI).
        """
        for param in self.clip_model.parameters():
            param.requires_grad = True
        log.info("Backbone unfrozen for fine-tuning")

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract normalized image embeddings from CLIP's image encoder.

        This is Step III of the methodology: "Extract features using CLIP"
        for the IMAGE modality.

        Args:
            images: Tensor of shape [N, 3, H, W]
                    Already preprocessed (normalized) by the dataloader.

        Returns:
            Tensor of shape [N, embedding_dim]
            L2-normalized (unit vectors on a hypersphere).
            Normalization is standard for CLIP — it makes cosine similarity
            equivalent to dot product, which speeds up computation.
        """
        images = images.to(self.device)

        if self._clip_type == "openai":
            features = self.clip_model.encode_image(images)
        else:
            features = self.clip_model.encode_image(images)

        # L2 normalize — standard CLIP practice
        features = features / features.norm(dim=-1, keepdim=True)
        return features.float()

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """
        Extract normalized text embeddings from CLIP's text encoder.

        This is Step III of the methodology: "Extract features using CLIP"
        for the TEXT modality.

        Args:
            texts: List of N strings (captions or demographic prompts)

        Returns:
            Tensor of shape [N, embedding_dim], L2-normalized
        """
        # Tokenize text — converts strings to integer token IDs
        tokens = self.tokenizer(texts)
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.to(self.device)

        if self._clip_type == "openai":
            features = self.clip_model.encode_text(tokens)
        else:
            features = self.clip_model.encode_text(tokens)

        features = features / features.norm(dim=-1, keepdim=True)
        return features.float()

    def encode_images_grad(self, images: torch.Tensor) -> torch.Tensor:
        """
        Same as encode_images() but WITH gradient tracking.
        Used during the training phase (Steps VI, VII) when we need
        gradients to flow back through the image encoder.
        """
        images = images.to(self.device)
        features = self.clip_model.encode_image(images)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.float()

    def encode_text_grad(self, texts: list[str]) -> torch.Tensor:
        """
        Same as encode_text() but WITH gradient tracking.
        Used during training.
        """
        tokens = self.tokenizer(texts)
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.to(self.device)
        features = self.clip_model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.float()

    @property
    def dtype(self):
        """Return the dtype of the model (float16 or float32)."""
        return next(self.clip_model.parameters()).dtype

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension (512 for ViT-B, 768 for ViT-L)."""
        return self.embedding_dim

    def __repr__(self):
        return (
            f"CLIPBackbone(\n"
            f"  model={self.model_name},\n"
            f"  embedding_dim={self.embedding_dim},\n"
            f"  device={self.device},\n"
            f"  clip_type={self._clip_type}\n"
            f")"
        )
