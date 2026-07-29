"""Iterative application of Step IV (subspace refit) + Step VIII (projection)."""
import torch
from omegaconf import OmegaConf
from models.fairclip import FairCLIP
from data.datasets import FaceDataset, collate_dict
from torch.utils.data import DataLoader
from evaluation.metrics.all_metrics import maxskew_ndkl_zhang, ZHANG_NEUTRAL_QUERIES

cfg = OmegaConf.load("configs/datasets/local_paths.yaml")
ATTR = "race"
m = FairCLIP(model_name="ViT-B/32", device="cuda")
m.load(f"results/checkpoints/fairface/ViT-B_32/{ATTR}/seed42/best_model.pt"); m.eval()
ds = FaceDataset(cfg.fairface.manifest, split="val", train=False)
ld = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_dict, num_workers=2)
img, labs = [], []
with torch.no_grad():
    for b in ld:
        img.append(m.backbone.encode_images(b["image"].to("cuda")).cpu())
        labs.append(b[ATTR].clone().detach())
img = torch.cat(img).float(); labs = torch.cat(labs)
img = img/img.norm(dim=-1, keepdim=True)
txt = m.encode_text(ZHANG_NEUTRAL_QUERIES).cpu().float()
txt = txt/txt.norm(dim=-1, keepdim=True)

def fit_subspace(E, labels, k):
    groups = sorted(labels.unique().tolist())
    cents = torch.stack([E[labels==g].mean(0) for g in groups])
    diffs = cents - cents.mean(0, keepdim=True)
    _,_,Vt = torch.linalg.svd(diffs, full_matrices=False)
    return Vt.T[:, :k]

def proj_out(E, B):
    d = E - (E@B)@B.T
    return d/d.norm(dim=-1, keepdim=True).clamp(min=1e-8)

ei, et = img.clone(), txt.clone()
print(f"{ATTR}: iterative Step IV+VIII rounds (k=6 per round)")
for r in range(1, 7):
    B = fit_subspace(ei, labs, k=6)
    ei = proj_out(ei, B); et = proj_out(et, B)
    ms, nd = maxskew_ndkl_zhang(et, ei, labs, k_skew=1000, k_ndkl=1000)
    print(f"  round {r}: MaxSkew={ms:.3f} NDKL={nd:.3f}")
print("Target: below 0.353 (Zhang). Single-round best was ~0.9")
