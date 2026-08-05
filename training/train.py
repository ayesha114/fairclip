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

def load_coco_captions(manifest_path, n=50000):
    """Load COCO captions for neutral contrastive training."""
    import pandas as pd
    try:
        df = pd.read_csv(manifest_path)
        captions = df["caption"].dropna().tolist()
        import random
        random.shuffle(captions)
        return captions[:n]
    except Exception:
        return None


# Demographic text prompts for each attribute
# Using attribute-specific prompts gives meaningful image-text pairs
ATTRIBUTE_PROMPTS = {
    "age": [
        "A photo of a 0-2 year old person",
        "A photo of a 3-9 year old person",
        "A photo of a 10-19 year old person",
        "A photo of a 20-29 year old person",
        "A photo of a 30-39 year old person",
        "A photo of a 40-49 year old person",
        "A photo of a 50-59 year old person",
        "A photo of a 60-69 year old person",
        "A photo of an elderly person",
    ],
    "gender": [
        "A photo of a Male person",
        "A photo of a Female person",
    ],
    "race": [
        "A photo of a person with light skin tone",
        "A photo of a person with dark skin tone",
        "A photo of a person with olive skin tone",
        "A photo of a person with yellow skin tone",
        "A photo of a person with tan skin tone",
        "A photo of a person with brown skin tone",
        "A photo of a person with warm skin tone",
    ],
}



def parse_args():
    parser = argparse.ArgumentParser(description="Train FairCLIP")
    parser.add_argument("--config",     default="configs/datasets/local_paths.yaml")
    parser.add_argument("--dataset",    default="fairface", choices=["fairface","utkface"])
    parser.add_argument("--attribute",  default="gender",
                        choices=["gender", "age", "race"],
                        help="Which demographic attribute to debias")
    parser.add_argument("--backbone",   default="ViT-B/32",
                        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-H/14"],
                        help="CLIP backbone to use")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lambda_fair",type=float, default=0.1,
                        help="Fairness loss weight (λ)")
    parser.add_argument("--lambda_retrieval", type=float, default=0.0,
                        help="Retrieval-skew loss weight")
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
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers",type=int,   default=4)
    parser.add_argument("--no_adaptive_temp", action="store_true")
    parser.add_argument("--no_procrustes", action="store_true")
    parser.add_argument("--no_train_proj", action="store_true")
    parser.add_argument("--no_subspace", action="store_true")
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
    """Map attribute name to manifest column name (DataFrame uses _idx suffix)."""
    return {"gender": "gender_idx", "age": "age_idx", "race": "race_idx"}[attribute]

def get_attribute_column_for(attribute: str, dataset: str) -> str:
    if attribute == "race" and dataset == "utkface":
        return "race_idx_7"
    return get_attribute_column(attribute)


def build_dataloaders(cfg, args):
    """Build train and val dataloaders for FairFace."""
    attr_col = get_attribute_column_for(args.attribute, args.dataset)

    _manifest = cfg[args.dataset].manifest
    log.info(f"Dataset: {args.dataset}  manifest: {_manifest}")
    train_ds = FaceDataset(
        _manifest,
        split="train",
        train=True,
        race_column=("race_idx_7" if args.dataset=="utkface" else "race_idx"),
    )
    val_ds = FaceDataset(
        _manifest,
        split="val",
        train=False,
        race_column=("race_idx_7" if args.dataset=="utkface" else "race_idx"),
    )

    # Use demographic-balanced sampler for training
    # This ensures each batch has equal representation of demographic groups
    # Critical for the fairness loss to work properly (Step VI)
    train_labels = train_ds.df[attr_col].values
    # Calculate optimal samples per group
    # More samples = less noisy fairness gradient estimates
    # Standard: batch_size / n_groups, minimum 14 for stable training
    n_groups = len(set(train_labels.tolist()))
    samples_per_group = max(14, args.batch_size // n_groups)
    effective_batch = samples_per_group * n_groups
    log.info(f"Using {samples_per_group} samples/group × {n_groups} groups = {effective_batch} effective batch size")

    sampler = DemographicBalancedSampler(
        train_labels,
        batch_size=effective_batch,
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

    attr_col = get_attribute_column_for(args.attribute, args.dataset)

    # Load data in larger batches for efficiency (no gradient needed here)
    setup_loader = DataLoader(
        train_ds,
        batch_size=128,
        shuffle=False,
        collate_fn=collate_dict,
        num_workers=args.num_workers,
    )

    all_image_embs = []
    all_labels = []

    # Ensure float32 for stable computation
    model.backbone.clip_model = model.backbone.clip_model.float()
    log.info("Extracting CLIP embeddings for bias subspace fitting...")
    # Use CPU for bias fitting to save GPU memory for training
    fitting_device = args.device
    for i, batch in enumerate(setup_loader):
        images = batch["image"].to(fitting_device)
        batch_key = args.attribute
        labels = batch[batch_key].clone().detach().to(args.device)

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
    # Free GPU memory held during fitting so training has room (critical for ViT-H/14)
    import gc as _gc
    _gc.collect()
    torch.cuda.empty_cache()
    log.info(f"GPU cache cleared after fitting. Free now: {(torch.cuda.mem_get_info()[0]/1e9):.2f} GB")
    # Free GPU memory held during fitting so training has room (critical for ViT-H/14)
    import gc as _gc
    _gc.collect()
    torch.cuda.empty_cache()
    log.info(f"GPU cache cleared after fitting. Free now: {(torch.cuda.mem_get_info()[0]/1e9):.2f} GB")
    # Free GPU cache after fitting
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    log.info(f"GPU memory after fitting: {torch.cuda.memory_allocated()/1e9:.2f}GB used")


# Mixed precision (FP16): halves memory so large backbones (ViT-L/14, H/14)
# fit on 8GB. Enabled automatically when CUDA is available.
import torch as _torch_amp
import os as _os
_USE_AMP = _torch_amp.cuda.is_available() and _os.environ.get('FAIRCLIP_NO_AMP','0') != '1'
_scaler = _torch_amp.cuda.amp.GradScaler(enabled=_USE_AMP)


def train_one_epoch(
    model: FairCLIP,
    train_loader,
    optimizer,
    args,
    epoch: int,
) -> dict:
    """Run one complete training epoch."""
    model.train()
    # backbone stays frozen except the 2 layers we put in the optimizer

    attr_col = get_attribute_column_for(args.attribute, args.dataset)

    total_loss = 0.0
    total_infonce = 0.0
    total_fair_img = 0.0
    total_fair_txt = 0.0
    n_batches = 0
    t0 = time.time()

    for step, batch in enumerate(train_loader):
        images = batch["image"].to(args.device)
        batch_key = args.attribute
        labels = batch[batch_key].clone().detach().to(args.device)

        # Build text prompts from demographic attribute
        # For FairFace: we pair each image with a demographic description
        # This creates image-text pairs for the contrastive loss
        # Build texts using demographic labels for meaningful contrastive pairs
        # EXCEPTION: for race, use neutral texts to avoid reinforcing stereotypes
        prompts = ATTRIBUTE_PROMPTS[args.attribute]
        if args.attribute in ("race", "age") and getattr(args, "_coco_captions", None):
            # Mixed source: even idx -> demographic prompt (class signal),
            # odd idx -> neutral COCO caption (neutral geometry). Deterministic.
            caps = args._coco_captions
            base = step * len(labels)
            texts = [prompts[min(l.item(), len(prompts)-1)] if i % 2 == 0
                     else caps[(base + i) % len(caps)]
                     for i, l in enumerate(labels)]
        else:
            texts = [prompts[min(l.item(), len(prompts)-1)] for l in labels]

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass — Steps III, VI, VII, VIII
        with torch.cuda.amp.autocast(enabled=_USE_AMP):
            loss, info = model.training_step(images, texts, labels)

        # Backward pass
        if _USE_AMP:
            _scaler.scale(loss).backward()
            _scaler.unscale_(optimizer)
        else:
            loss.backward()

        # Clip gradients BEFORE optimizer step (critical for CLIP fine-tuning)
        torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group['params']],
            max_norm=0.1  # tight clipping for CLIP stability
        )

        if _USE_AMP:
            _scaler.step(optimizer)
            _scaler.update()
        else:
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
    attr_col = get_attribute_column_for(args.attribute, args.dataset)

    total_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        images = batch["image"].to(args.device)
        batch_key = args.attribute
        labels = batch[batch_key].clone().detach().to(args.device)

        # Build texts using demographic labels for meaningful contrastive pairs
        # EXCEPTION: for race, use neutral texts to avoid reinforcing stereotypes
        prompts = ATTRIBUTE_PROMPTS[args.attribute]
        texts = [prompts[min(l.item(), len(prompts)-1)] for l in labels]

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
    # (ablation attributes set after construction below)
    model = FairCLIP(
        model_name=args.backbone,
        device=args.device,
        n_bias_directions=args.n_bias_dirs,
        lambda_fair_image=args.lambda_fair,
        lambda_fair_text=args.lambda_fair,
        lambda_retrieval=args.lambda_retrieval,
        tau_base=args.tau_base,
        alpha_temperature=args.alpha_temp,
    )

    # Phase 1: Fit bias subspace (Steps IV + V)
    model.ablate_adaptive_temp = args.no_adaptive_temp
    model.ablate_procrustes = args.no_procrustes
    model.ablate_train_proj = args.no_train_proj
    if args.no_subspace:
        model.ablate_train_proj = True  # no subspace implies no projection
        model._bias_subspace_fitted = True  # skip fitting requirement
        import torch as _t
        d = model.get_embedding_dim()
        _pl = _t.randn(d,1); _pl = _pl / _pl.norm()
        model.bias_remover.set_bias_subspace(_pl.clone(), _pl.clone())
        model.backbone.clip_model = model.backbone.clip_model.float()  # match fit_bias_subspace: ensure float32 for stability
    else:
        fit_bias_subspace(model, train_ds, args)

    # Only fine-tune the last 2 transformer blocks + projection
    # Fine-tuning ALL of CLIP causes gradient explosion
    # This is standard practice for CLIP fine-tuning
    trainable_params = []
    
    # Visual transformer - last 2 blocks only
    visual = model.backbone.clip_model.visual
    if hasattr(visual, 'transformer'):
        blocks = visual.transformer.resblocks
        for block in list(blocks)[-2:]:
            trainable_params.extend(block.parameters())
    if hasattr(visual, 'proj') and visual.proj is not None:
        trainable_params.append(visual.proj)
    
    # Text transformer - last 2 blocks only  
    transformer = model.backbone.clip_model.transformer
    if hasattr(transformer, 'resblocks'):
        blocks = transformer.resblocks
        for block in list(blocks)[-2:]:
            trainable_params.extend(block.parameters())
    
    # Text projection
    if hasattr(model.backbone.clip_model, 'text_projection'):
        tp = model.backbone.clip_model.text_projection
        if tp is not None:
            trainable_params.append(tp)

    # Freeze everything first, then unfreeze ONLY the trainable params.
    # This keeps memory low (only these layers build gradients) but ensures
    # the loss has a grad path (fixes "does not require grad").
    for p in model.backbone.clip_model.parameters():
        p.requires_grad = False
    for p in trainable_params:
        p.requires_grad = True
    n_params = sum(p.numel() for p in trainable_params)
    log.info(f"Trainable parameters: {n_params:,} (last 2 layers only)")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=0.01,
    )

    # Scheduler — reduce LR on plateau
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # Output directory
    out_dir = Path(args.output_dir) / args.dataset / args.backbone.replace("/", "_") / args.attribute / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Phase 2: Training loop (Steps VI + VII + VIII)
    best_val_loss = float("inf")
    no_improve_count = 0  # Early stopping counter
    # Load COCO captions for race training (neutral contrastive pairs)
    if args.attribute in ("race", "age"):
        coco_manifest = cfg.coco.get("manifest_train", "results/manifests/coco_train.csv")
        captions = load_coco_captions(coco_manifest)
        if captions:
            args._coco_captions = captions
            log.info(f"Loaded {len(captions)} COCO captions for race contrastive training")
        else:
            args._coco_captions = None
            log.warning("COCO captions not found — falling back to demographic prompts")
    else:
        args._coco_captions = None

    log.info("Starting training...")


    # =========================================================================
    # DISASTER RECOVERY: Auto-resume from last checkpoint if it exists
    # If training stops due to power cut / crash, just re-run the same command
    # =========================================================================
    start_epoch = 1
    latest_ckpt = out_dir / "latest.pt"

    if latest_ckpt.exists():
        log.info(f"Found checkpoint: {latest_ckpt} — resuming training...")
        ckpt = torch.load(str(latest_ckpt), map_location=args.device)
        start_epoch = ckpt["epoch"] + 1

        # Restore CLIP weights
        model.backbone.clip_model.load_state_dict(ckpt["clip_state_dict"])
        model.backbone.clip_model = model.backbone.clip_model.float()

        # Restore bias subspace
        state = ckpt["model_state"]
        model.bias_discoverer.image_bias_directions = state["bias_discoverer_image"]
        model.bias_discoverer.text_bias_directions = state["bias_discoverer_text"]
        model.procrustes.rotation_matrix = state["procrustes_rotation"]
        model._bias_subspace_fitted = state["bias_fitted"]
        model.bias_remover.set_bias_subspace(
            state["bias_discoverer_image"],
            state["bias_discoverer_text"],
        )

        # Restore optimizer and scheduler
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        log.info(f"Resumed from epoch {ckpt['epoch']} — continuing from epoch {start_epoch}")
    else:
        log.info("No checkpoint found — starting fresh.")

    for epoch in range(start_epoch, args.epochs + 1):
        log.info(f"\nEpoch {epoch}/{args.epochs}")

        # Reset temperature history each epoch
        model.temperature_controller.reset_history()

        # Refit bias subspace every 5 epochs to keep directions fresh
        # This fixes the stale bias direction problem
        if epoch % 5 == 0:
            log.info(f"Refitting bias subspace at epoch {epoch}...")
            fit_bias_subspace(model, train_ds, args)

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

        # Save full training state every epoch (disaster recovery)
        # Save only latest.pt (for resume) - not individual epochs (saves disk space)
        latest_path = out_dir / "latest.pt"
        torch.save({
            "epoch": epoch,
            "model_state": {
                "bias_discoverer_image": model.bias_discoverer.image_bias_directions,
                "bias_discoverer_text": model.bias_discoverer.text_bias_directions,
                "procrustes_rotation": model.procrustes.rotation_matrix,
                "bias_fitted": model._bias_subspace_fitted,
            },
            "clip_state_dict": model.backbone.clip_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "args": vars(args),
        }, latest_path)

        log.info(f"  Checkpoint saved: latest.pt (epoch {epoch})")

        # Save best model separately
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            import shutil
            shutil.copy(str(latest_path), str(out_dir / "best_model.pt"))
            log.info(f"  New best model saved (epoch {epoch})")
            no_improve_count = 0  # Reset counter on improvement
        else:
            no_improve_count += 1
            log.info(f"  No improvement for {no_improve_count}/{args.patience} epochs")
            if no_improve_count >= args.patience:
                log.info(f"Early stopping triggered at epoch {epoch}!")
                log.info(f"Best val loss: {best_val_loss:.4f} (epoch {epoch - no_improve_count})")
                break

    log.info("\nTraining complete!")
    log.info(f"Best val loss: {best_val_loss:.4f}")
    log.info(f"Checkpoints saved to: {out_dir}")


if __name__ == "__main__":
    main()
