"""
=============================================================================
FairCLIP — Training Loop
=============================================================================
This runs the complete training pipeline.

How to run:
    cd ~/thesis
    python -m training.train \
        --attribute gender \
        --backbone ViT-B/32 \
        --epochs 10 \
        --batch_size 64 \
        --lambda_fair 0.1

What happens:
    1. Load FairFace dataset
    2. Load pretrained CLIP (Step III)
    3. Extract embeddings and fit bias subspace (Steps IV, V)
    4. Train with fairness-aware loss (Steps VI, VII, VIII)
    5. Save checkpoints every epoch
    6. Log all metrics to console (and W&B if enabled)
=============================================================================
"""

import argparse
import logging
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.fairclip import FairCLIP
from data.datasets import FaceDataset, collate_dict
from data.balanced_sampler import DemographicBalancedSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")


def parse_args():
    parser = argparse.ArgumentParser(description="Train FairCLIP")
    parser.add_argument("--config",     default="configs/datasets/local_paths.yaml")
    parser.add_argument("--attribute",  default="gender",
                        choices=["gender", "age", "race"],
                        help="Which demographic attribute to debias")
    parser.add_argument("--backbone",   default="ViT-B/32",
                        choices=["ViT-B/32", "ViT-B/16"],
                        help="CLIP backbone to use")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lambda_fair",type=float, default=0.1,
                        help="Fairness loss weight (λ)")
    parser.add_argument("--n_bias_dirs",type=int,   default=5,
                        help="Number of PCA bias directions")
    parser.add_argument("--tau_base",   type=float, default=0.07,
                        help="Base temperature for contrastive loss")
    parser.add_argument("--alpha_temp", type=float, default=0.5,
                        help="Adaptive temperature sensitivity")
    parser.add_argument("--lr",         type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--output_dir", default="results/checkpoints")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers",type=int,   default=4)
    parser.add_argument("--log_every",  type=int,   default=50,
                        help="Log training stats every N steps")
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np, random
    np.random.seed(seed)
    random.seed(seed)


def get_attribute_column(attribute: str) -> str:
    """Map attribute name to manifest column name."""
    return {"gender": "gender_idx", "age": "age_idx", "race": "race_idx"}[attribute]


def build_dataloaders(cfg, args):
    """Build train and val dataloaders for FairFace."""
    attr_col = get_attribute_column(args.attribute)

    train_ds = FaceDataset(
        cfg.fairface.manifest,
        split="train",
        train=True,
        race_column=attr_col if args.attribute == "race" else "race_idx",
    )
    val_ds = FaceDataset(
        cfg.fairface.manifest,
        split="val",
        train=False,
    )

    # Use demographic-balanced sampler for training
    # This ensures each batch has equal representation of demographic groups
    # Critical for the fairness loss to work properly (Step VI)
    train_labels = train_ds.df[attr_col].values
    sampler = DemographicBalancedSampler(
        train_labels,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_sampler=sampler.batches(),
        collate_fn=collate_dict,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_dict,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    log.info(f"Train: {len(train_ds)} samples")
    log.info(f"Val:   {len(val_ds)} samples")
    return train_loader, val_loader, train_ds


@torch.no_grad()
def fit_bias_subspace(model: FairCLIP, train_ds, args):
    """
    Extract embeddings for the full training set and fit the bias subspace.

    This is a one-time setup step before training begins.
    We use the ENTIRE training set so the bias directions are accurate.

    For large datasets this takes a few minutes — it only runs once.
    """
    log.info("Fitting bias subspace on training data (Steps IV + V)...")
    log.info("This runs once before training starts.")

    attr_col = get_attribute_column(args.attribute)

    # Load data in larger batches for efficiency (no gradient needed here)
    setup_loader = DataLoader(
        train_ds,
        batch_size=256,
        shuffle=False,
        collate_fn=collate_dict,
        num_workers=args.num_workers,
    )

    all_image_embs = []
    all_labels = []

    log.info("Extracting CLIP embeddings for bias subspace fitting...")
    for i, batch in enumerate(setup_loader):
        images = batch["image"].to(args.device)
        labels = torch.tensor(batch[attr_col]).to(args.device)

        image_embs = model.backbone.encode_images(images)
        all_image_embs.append(image_embs.cpu())
        all_labels.append(labels.cpu())

        if (i + 1) % 20 == 0:
            log.info(f"  Processed {(i+1)*256} / {len(train_ds)} samples")

    all_image_embs = torch.cat(all_image_embs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # For text embeddings, we use demographic prompts
    # This captures how CLIP represents demographic concepts in text
    text_prompts = {
        "gender": ["A photo of a Male person", "A photo of a Female person"],
        "age": [f"A photo of a {age} year old person" for age in
                ["young", "teenage", "adult", "middle-aged", "senior"]],
        "race": ["A photo of a White person", "A photo of a Black person",
                 "A photo of a Latino person", "A photo of an East Asian person",
                 "A photo of a Southeast Asian person", "A photo of an Indian person",
                 "A photo of a Middle Eastern person"],
    }
    prompts = text_prompts[args.attribute]
    text_embs = model.backbone.encode_text(prompts)
    text_labels = torch.arange(len(prompts))

    # Fit bias subspace (Steps IV + V)
    model.fit_bias_subspace(
        all_image_embs,
        text_embs.cpu(),
        all_labels,
        attribute=args.attribute,
    )
    log.info("Bias subspace fitted successfully")


def train_one_epoch(
    model: FairCLIP,
    train_loader,
    optimizer,
    args,
    epoch: int,
) -> dict:
    """Run one complete training epoch."""
    model.train()
    model.backbone.unfreeze_backbone()  # Allow CLIP weights to update

    attr_col = get_attribute_column(args.attribute)

    total_loss = 0.0
    total_infonce = 0.0
    total_fair_img = 0.0
    total_fair_txt = 0.0
    n_batches = 0
    t0 = time.time()

    for step, batch in enumerate(train_loader):
        images = batch["image"].to(args.device)
        labels = torch.tensor(batch[attr_col]).to(args.device)

        # Build text prompts from demographic attribute
        # For FairFace: we pair each image with a demographic description
        # This creates image-text pairs for the contrastive loss
        attr_strs = batch.get(f"{args.attribute}_str", None)
        if attr_strs is not None:
            texts = [f"A photo of a {s} person" for s in attr_strs]
        else:
            texts = [f"A photo of a person"] * len(images)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass — Steps III, VI, VII, VIII
        loss, info = model.training_step(images, texts, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping prevents training instability
        torch.nn.utils.clip_grad_norm_(
            model.backbone.clip_model.parameters(), max_norm=1.0
        )

        optimizer.step()

        # Accumulate stats
        total_loss += info["total"]
        total_infonce += info["infonce"]
        total_fair_img += info["fairness_image"]
        total_fair_txt += info["fairness_text"]
        n_batches += 1

        # Log every N steps
        if (step + 1) % args.log_every == 0:
            elapsed = time.time() - t0
            log.info(
                f"Epoch {epoch} | Step {step+1}/{len(train_loader)} | "
                f"Loss: {info['total']:.4f} | "
                f"InfoNCE: {info['infonce']:.4f} | "
                f"Fair_img: {info['fairness_image']:.4f} | "
                f"Fair_txt: {info['fairness_text']:.4f} | "
                f"τ: {info['temperature']:.4f} | "
                f"Time: {elapsed:.1f}s"
            )
            t0 = time.time()

    return {
        "loss": total_loss / n_batches,
        "infonce": total_infonce / n_batches,
        "fairness_image": total_fair_img / n_batches,
        "fairness_text": total_fair_txt / n_batches,
    }


@torch.no_grad()
def validate(model: FairCLIP, val_loader, args) -> dict:
    """Run validation — compute loss on val set."""
    model.eval()
    attr_col = get_attribute_column(args.attribute)

    total_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        images = batch["image"].to(args.device)
        labels = torch.tensor(batch[attr_col]).to(args.device)

        attr_strs = batch.get(f"{args.attribute}_str", None)
        if attr_strs is not None:
            texts = [f"A photo of a {s} person" for s in attr_strs]
        else:
            texts = [f"A photo of a person"] * len(images)

        # Use training_step in eval mode
        loss, info = model.training_step(images, texts, labels)
        total_loss += info["total"]
        n_batches += 1

    return {"val_loss": total_loss / n_batches}


def main():
    args = parse_args()
    set_seed(args.seed)

    log.info("=" * 60)
    log.info("FairCLIP Training")
    log.info(f"Attribute: {args.attribute}")
    log.info(f"Backbone:  {args.backbone}")
    log.info(f"Epochs:    {args.epochs}")
    log.info(f"Batch:     {args.batch_size}")
    log.info(f"Lambda:    {args.lambda_fair}")
    log.info(f"Device:    {args.device}")
    log.info(f"Seed:      {args.seed}")
    log.info("=" * 60)

    # Load config
    cfg = OmegaConf.load(args.config)

    # Build dataloaders
    train_loader, val_loader, train_ds = build_dataloaders(cfg, args)

    # Initialize FairCLIP model
    model = FairCLIP(
        model_name=args.backbone,
        device=args.device,
        n_bias_directions=args.n_bias_dirs,
        lambda_fair_image=args.lambda_fair,
        lambda_fair_text=args.lambda_fair,
        tau_base=args.tau_base,
        alpha_temperature=args.alpha_temp,
    )

    # Phase 1: Fit bias subspace (Steps IV + V)
    fit_bias_subspace(model, train_ds, args)

    # Optimizer — only update CLIP backbone parameters
    optimizer = torch.optim.AdamW(
        model.backbone.clip_model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    # Scheduler — reduce LR on plateau
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # Output directory
    out_dir = Path(args.output_dir) / args.backbone.replace("/", "_") / args.attribute
    out_dir.mkdir(parents=True, exist_ok=True)

    # Phase 2: Training loop (Steps VI + VII + VIII)
    best_val_loss = float("inf")
    log.info("Starting training...")

    for epoch in range(1, args.epochs + 1):
        log.info(f"\nEpoch {epoch}/{args.epochs}")

        # Reset temperature history each epoch
        model.temperature_controller.reset_history()

        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, args, epoch)

        # Validate
        val_metrics = validate(model, val_loader, args)

        scheduler.step()

        # Log epoch summary
        log.info(
            f"Epoch {epoch} Summary: "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['val_loss']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        temp_stats = model.temperature_controller.get_current_stats()
        log.info(
            f"  Temperature: mean={temp_stats.get('temperature_mean', 0):.4f} "
            f"min={temp_stats.get('temperature_min', 0):.4f} "
            f"max={temp_stats.get('temperature_max', 0):.4f}"
        )

        # Save best checkpoint
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            ckpt_path = out_dir / "best_model.pt"
            model.save(str(ckpt_path))
            log.info(f"  New best model saved: {ckpt_path}")

        # Save latest checkpoint every epoch
        latest_path = out_dir / f"epoch_{epoch:03d}.pt"
        model.save(str(latest_path))

    log.info("\nTraining complete!")
    log.info(f"Best val loss: {best_val_loss:.4f}")
    log.info(f"Checkpoints saved to: {out_dir}")


if __name__ == "__main__":
    main()
