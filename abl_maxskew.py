"""MaxSkew (iterative protocol) for each ablation checkpoint."""
import torch, os, glob
from omegaconf import OmegaConf
from models.fairclip import FairCLIP
from data.datasets import FaceDataset, collate_dict
from torch.utils.data import DataLoader
from evaluation.metrics.all_metrics import maxskew_ndkl_zhang, ZHANG_NEUTRAL_QUERIES
cfg = OmegaConf.load("configs/datasets/local_paths.yaml")
def fit(E, labels, k=6):
    g = sorted(labels.unique().tolist())
    c = torch.stack([E[labels==x].mean(0) for x in g])
    _,_,Vt = torch.linalg.svd(c - c.mean(0, keepdim=True), full_matrices=False)
    return Vt.T[:, :k]
def proj(E, B):
    d = E - (E@B)@B.T
    return d/d.norm(dim=-1, keepdim=True).clamp(min=1e-8)
for name in ["no_fairloss","no_temp","no_trainproj","no_procrustes","no_subspace"]:
    cks = glob.glob(f"results/checkpoints_abl_{name}/**/best_model.pt", recursive=True)
    if not cks: print(f"{name}: no checkpoint"); continue
    m = FairCLIP(model_name="ViT-B/32", device="cuda"); m.load(cks[0]); m.eval()
    ds = FaceDataset(cfg.fairface.manifest, split="val", train=False)
    ld = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_dict, num_workers=2)
    img, labs = [], []
    with torch.no_grad():
        for b in ld:
            img.append(m.backbone.encode_images(b["image"].to("cuda")).cpu())
            labs.append(b["race"].clone().detach())
    img = torch.cat(img).float(); labs = torch.cat(labs)
    img = img/img.norm(dim=-1, keepdim=True)
    txt = m.encode_text(ZHANG_NEUTRAL_QUERIES).cpu().float()
    txt = txt/txt.norm(dim=-1, keepdim=True)
    ei, et = img.clone(), txt.clone()
    best = 9e9
    for r in range(1, 7):
        B = fit(ei, labs); ei = proj(ei, B); et = proj(et, B)
        ms, nd = maxskew_ndkl_zhang(et, ei, labs, k_skew=1000, k_ndkl=1000)
        best = min(best, ms)
    print(f"{name}: best iterative MaxSkew={best:.3f}  (full model: 0.214)")
    del m; torch.cuda.empty_cache()
