"""Leakage check: fit projection subspace on TRAIN, apply to VAL (no refit on val).
If MaxSkew stays low -> clean (generalizes). If it jumps up -> refitting on val
was doing the work (leakage to fix)."""
import torch
from omegaconf import OmegaConf
from models.fairclip import FairCLIP
from data.datasets import FaceDataset, collate_dict
from torch.utils.data import DataLoader
from evaluation.metrics.all_metrics import maxskew_ndkl_zhang, ZHANG_NEUTRAL_QUERIES
torch.set_num_threads(4)
cfg=OmegaConf.load("configs/datasets/local_paths.yaml")
DEV="cpu"; ATTR="race"
m=FairCLIP(model_name="ViT-B/32",device=DEV)
m.load(f"results/checkpoints/fairface/ViT-B_32/{ATTR}/seed42/best_model.pt"); m.eval()
def embed(split, n):
    ds=FaceDataset(cfg.fairface.manifest,split=split,train=False)
    ld=DataLoader(ds,batch_size=32,shuffle=False,collate_fn=collate_dict,num_workers=2)
    E,L=[],[]
    with torch.no_grad():
        for b in ld:
            E.append(m.backbone.encode_images(b["image"].to(DEV)).cpu()); L.append(b[ATTR].clone().detach())
            if sum(x.shape[0] for x in E)>=n: break
    E=torch.cat(E)[:n].float(); L=torch.cat(L)[:n]; return E/E.norm(dim=-1,keepdim=True), L
print("loading train + val subsets...", flush=True)
tr,trl=embed("train",2000); va,val=embed("val",2000)
neu=m.encode_text(ZHANG_NEUTRAL_QUERIES).cpu().float(); neu=neu/neu.norm(dim=-1,keepdim=True)
def fit(E,l,k=6):
    g=sorted(l.unique().tolist()); c=torch.stack([E[l==x].mean(0) for x in g])
    _,_,Vt=torch.linalg.svd(c-c.mean(0,keepdim=True),full_matrices=False); return Vt.T[:,:k]
def proj(E,B):
    d=E-(E@B)@B.T; return d/d.norm(dim=-1,keepdim=True).clamp(min=1e-8)
# METHOD A (current): refit on val each round
eva,eneu=va.clone(),neu.clone()
for _ in range(8):
    B=fit(eva,val); eva=proj(eva,B); eneu=proj(eneu,B)
msA,ndA=maxskew_ndkl_zhang(eneu,eva,val,k_skew=1000,k_ndkl=1000)
# METHOD B (clean): fit subspace on TRAIN once, apply to val (no val refit)
evb,eneu2=va.clone(),neu.clone()
etr=tr.clone()
for _ in range(8):
    B=fit(etr,trl); etr=proj(etr,B); evb=proj(evb,B); eneu2=proj(eneu2,B)
msB,ndB=maxskew_ndkl_zhang(eneu2,evb,val,k_skew=1000,k_ndkl=1000)
print(f"\nMETHOD A (refit on val, current):  MaxSkew={msA:.3f} NDKL={ndA:.3f}")
print(f"METHOD B (fit on train, clean):    MaxSkew={msB:.3f} NDKL={ndB:.3f}")
print("\nIf B is close to A -> CLEAN (no leakage, generalizes).")
print("If B is much worse than A -> refitting on val was the trick (must fix).")
