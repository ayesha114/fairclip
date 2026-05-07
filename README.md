# FairCLIP

> Reducing Social Bias in Vision-Language Models via Training-Time Joint Debiasing

This repository is the official implementation of FairCLIP, a training-time fairness-aware framework for CLIP that prevents bias from forming during contrastive learning. The method combines four novel components: PCA-based bias subspace discovery, orthogonal Procrustes cross-modal alignment, fairness-aware contrastive loss, and adaptive temperature scaling.

**MS Thesis** — National University of Computer and Emerging Sciences  
**Author**: Ayesha Kalsoom (25I-7805) | **Supervisor**: Dr. Afia Zafar

---

## Status

This is **Milestone 1 of 8** — the data pipeline. See `docs/build_status.md` for what is implemented and what is coming next.

| Milestone | Status |
|---|---|
| M1 — Foundation + Data | ✅ Complete |
| M2 — Bias subspace + Procrustes | ⏳ Coming next |
| M3 — Fairness loss + adaptive τ | ⏳ |
| M4 — Integration + training loop | ⏳ |
| M5 — Baseline re-implementations | ⏳ |
| M6 — Evaluation & metrics | ⏳ |
| M7 — Explainability + analysis | ⏳ |
| M8 — HPC scripts + paper template | ⏳ |

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/fairclip.git
cd fairclip
chmod +x setup_local.sh
./setup_local.sh
conda activate fairclip
```

### 2. Configure dataset paths

Edit `configs/datasets/local_paths.yaml` and replace each placeholder path with the location of the dataset on your machine:

```yaml
fairface:
  root: "/home/yourname/datasets/fairface"   # ← put your real path here
```

### 3. Build manifests

Each `prepare_*.py` script reads the raw dataset and writes a clean manifest CSV that the dataloader will consume.

```bash
python -m data.prepare_fairface    # primary fairness benchmark
python -m data.prepare_utkface     # cross-dataset generalization
python -m data.prepare_coco        # retrieval evaluation
python -m data.prepare_facet       # out-of-domain fairness eval
```

### 4. Verify everything works

```bash
python -m data.verify_data
```

This loads each dataset, pulls a few samples, and tests the demographic-balanced sampler. If anything fails, the error message tells you exactly what to fix.

### 5. Run unit tests

```bash
pytest tests/ -v
```

All tests use synthetic data, so they run without any datasets being present.

---

## Datasets

| Dataset | Role | License | Source |
|---|---|---|---|
| FairFace | Primary fairness training/eval | CC BY 4.0 | [GitHub](https://github.com/joojs/fairface) |
| UTKFace | Cross-dataset generalization | Research only | [Project page](https://susanqq.github.io/UTKFace/) |
| COCO Captions | Retrieval evaluation | CC BY 4.0 | [cocodataset.org](https://cocodataset.org) |
| FACET | Out-of-domain fairness | Research only | [facet.metademolab.com](https://facet.metademolab.com) |
| Flickr30k | Retrieval (optional) | Research only | [hockenmaier.cs.illinois.edu](http://hockenmaier.cs.illinois.edu/DenotationGraph/) |
| ImageNet val | Zero-shot (optional) | ImageNet license | [image-net.org](https://www.image-net.org) |

You **download these yourself** and point the config at them. We do not redistribute any data.

---

## Repository Structure

```
fairclip/
├── configs/                Hyperparameters as YAML
│   ├── datasets/local_paths.yaml    ← THE FILE YOU EDIT
│   ├── backbones/                   ViT-B/32, ViT-B/16, ViT-L/14
│   ├── attributes/                  gender / age / race / universal
│   └── ablations/                   for ablation studies
├── data/                   Dataset prep + PyTorch loaders
├── models/                 [M2-M3] FairCLIP components
├── losses/                 [M3] fairness regularizer + total loss
├── baselines/              [M5] Zhang, CLIP-clip, Biased-prompts, BendVLM
├── training/               [M4] training loop + checkpointing
├── evaluation/             [M6] metrics and eval pipelines
├── explainability/         [M7] PCA viz, Grad-CAM, t-SNE
├── analysis/               [M7] table & figure generators
├── tests/                  unit tests (synthetic data — no downloads needed)
├── scripts/                CLI entrypoints + SLURM scripts for HPC
├── paper/                  LaTeX manuscript
└── results/                outputs land here (gitignored)
```

---

## Method (Mapped to Proposal)

| Step in proposal | File | Status |
|---|---|---|
| I. Data collection (FairFace, COCO, FACET) | `data/prepare_*.py` | ✅ |
| II. Preprocessing | `data/datasets.py` | ✅ |
| III. Feature extraction (CLIP) | `models/clip_backbone.py` | M2 |
| IV. Bias subspace (PCA/SVD) | `models/bias_subspace.py` | M2 |
| V. Cross-modal alignment (Procrustes) | `models/procrustes_alignment.py` | M2 |
| VI. Fairness-aware contrastive learning | `losses/total_loss.py` | M3 |
| VII. Adaptive temperature scaling | `models/adaptive_temperature.py` | M3 |
| VIII. Bias removal | `models/bias_removal.py` | M3 |
| IX. Evaluation | `evaluation/` | M6 |

---

## Citation

If you find this work useful, please cite:

```bibtex
@mastersthesis{kalsoom2026fairclip,
  title  = {FairCLIP: Reducing Social Bias in Vision-Language Models},
  author = {Kalsoom, Ayesha},
  year   = {2026},
  school = {National University of Computer and Emerging Sciences}
}
```

A peer-reviewed publication is forthcoming.

---

## License

Code: MIT (see `LICENSE`).  
Datasets retain their respective licenses — please read them before use.

---

## Acknowledgments

This work builds on:
- **Zhang et al. CVPR 2025** — Joint Vision-Language Social Bias Removal for CLIP (baseline)
- **OpenAI CLIP** and **OpenCLIP** for the pretrained backbones
- The FairFace, UTKFace, COCO, and FACET dataset creators
