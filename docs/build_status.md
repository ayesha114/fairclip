# Build Status

This file tracks what's implemented vs. what's coming. Show this to your supervisor.

## Milestone 1 — Foundation + Data ✅ COMPLETE

**Delivered files:**

| File | Purpose | LOC |
|---|---|---|
| `requirements.txt` | Pinned Python dependencies | 63 |
| `setup_local.sh` | One-command Ubuntu setup | 122 |
| `configs/datasets/local_paths.yaml` | Single edit point for dataset paths | 97 |
| `data/prepare_fairface.py` | FairFace manifest builder | 283 |
| `data/prepare_utkface.py` | UTKFace manifest builder | 200 |
| `data/prepare_coco.py` | COCO Captions manifest builder | 145 |
| `data/prepare_facet.py` | FACET manifest builder | 175 |
| `data/datasets.py` | PyTorch Dataset classes | 320 |
| `data/balanced_sampler.py` | Demographic-balanced sampler | 220 |
| `data/verify_data.py` | One-command data sanity check | 145 |
| `data/__init__.py` | Package init | 30 |
| `tests/test_dataloader.py` | 18 unit tests, all passing | 245 |
| `README.md` | GitHub landing page | 180 |
| `.gitignore` | Keeps repo clean | 75 |
| `LICENSE` | MIT | 21 |

**Tests passing:** 18/18 (uses synthetic data, no downloads required to run)

**What you can do RIGHT NOW with M1:**
- Set up the Ubuntu environment with one command
- Point at your local datasets via a single config file
- Build manifests for FairFace, UTKFace, COCO, FACET
- Load data with PyTorch DataLoader using the balanced sampler
- Push to GitHub as a clean, professional repo

## Milestone 2 — Bias Subspace + Procrustes (NEXT)

Will add:
- `models/clip_backbone.py` — switchable ViT-B/32, ViT-B/16, ViT-L/14
- `models/bias_subspace.py` — Step IV: PCA/SVD on group centroids
- `models/procrustes_alignment.py` — Step V: closed-form orthogonal alignment
- `tests/test_bias_subspace.py`
- `tests/test_procrustes.py`
- Math notes in `docs/method_formalization.md` (theory section drafts)

## Milestone 3 — Fairness Loss + Adaptive τ

Will add:
- `losses/contrastive.py` — InfoNCE
- `losses/fairness_regularizer.py` — Step VI: group-variance penalty
- `losses/total_loss.py` — combined objective
- `models/adaptive_temperature.py` — Step VII
- `models/bias_removal.py` — Step VIII
- Tests for each

## Milestone 4 — Integration + Training

Will add:
- `models/fairclip.py` — full FairCLIP module
- `training/trainer.py` — main training loop with AMP
- `training/checkpoint.py` — save/resume
- `training/train.py` — CLI entrypoint

## Milestone 5 — Baselines

Will add re-implementations of Zhang 2025, CLIP-clip, Biased-prompts, BendVLM
for direct apples-to-apples comparison.

## Milestone 6 — Evaluation

Will add all metrics (MaxSkew, NDKL, ABLE, DPG, EOD), retrieval, classification,
and cross-dataset eval.

## Milestone 7 — Explainability + Analysis

Will add Grad-CAM, t-SNE, PCA visualization, and table/figure generators
for the paper.

## Milestone 8 — HPC Scripts + Paper

Will add SLURM scripts for the university HPC and the LaTeX paper template.
