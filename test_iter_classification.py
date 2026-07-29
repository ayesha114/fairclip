"""Verify zero-shot classification on iteratively-projected embeddings (round 5)."""
import torch
from omegaconf import OmegaConf
from models.fairclip import FairCLIP
from data.datasets import FaceDataset, collate_dict
from torch.utils.data import DataLoader
from evaluation.metrics.all_metrics import maxskew_ndkl_zhang, ZHANG_NEUTRAL_QUERIES

cfg = OmegaConf.load("configs/datasets/local_paths.yaml")
ATTR = "race"
RACE_PROMPTS = ["a photo of a White person","a photo of a Black person",
 "a photo of a Latino Hispanic person","a photo of an East Asian person",
 "a photo of a Southeast Asian person","a photo of an Indian person",
 "a photo of a Middle Eastern person"]
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
cls_txt = m.encode_text(RACE_PROMPTS).cpu().float()
cls_txt = cls_txt/cls_txt.norm(dim=-1, keepdim=True)

def fit_subspace(E, labels, k=6):
    groups = sorted(labels.unique().tolist())
    cents = torch.stack([E[labels==g].mean(0) for g in groups])
    diffs = cents - cents.mean(0, keepdim=True)
    _,_,Vt = torch.linalg.svd(diffs, full_matrices=False)
    return Vt.T[:, :k]

def proj_out(E, B):
    d = E - (E@B)@B.T
    return d/d.norm(dim=-1, keepdim=True).clamp(min=1e-8)

ei, ct = img.clone(), cls_txt.clone()
for r in range(1, 6):
    B = fit_subspace(ei, labs)
    ei = proj_out(ei, B); ct = proj_out(ct, B)
    pred = (ei @ ct.T).argmax(dim=1)
    acc = (pred == labs).float().mean().item()
    # DPG one-vs-rest
    import numpy as np
    dpgs=[]
    for c in range(len(RACE_PROMPTS)):
        rates=[]
        for g in sorted(labs.unique().tolist()):
            mask = labs==g
            rates.append(((pred[mask]==c).float().mean()).item())
        dpgs.append(max(rates)-min(rates))
    print(f"round {r}: Acc={acc:.3f} DPG={np.mean(dpgs):.3f}")
print("Need: Acc>0.6 and DPG<0.45 at the round where MaxSkew<0.353 (round 2+)")
