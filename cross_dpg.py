"""Cross-dataset: subspace fit on FairFace train, applied to UTKFace val. No training."""
import argparse, torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from models.fairclip import FairCLIP
from data.datasets import FaceDataset, collate_dict
from evaluation.metrics.all_metrics import (
    compute_dpg_occupation, compute_eod_occupation, compute_rbs, OCCUPATION_PROMPTS)

def encode(m, manifest, attr, split, dev, bb, race_col, cap=None):
    ds = FaceDataset(manifest, split=split, train=False, race_column=race_col)
    ebs = 16 if bb in ("ViT-L/14","ViT-H/14") else 64
    ld = DataLoader(ds, batch_size=ebs, shuffle=False, collate_fn=collate_dict, num_workers=2)
    E,L=[],[]
    with torch.no_grad():
        for b in ld:
            E.append(m.backbone.encode_images(b["image"].to(dev)).cpu())
            L.append(b[attr].clone().detach())
            if cap and sum(x.shape[0] for x in E)>=cap: break
    E=torch.cat(E).float(); L=torch.cat(L)
    if cap: E,L=E[:cap],L[:cap]
    return E/E.norm(dim=-1,keepdim=True), L

def fit_subspace(embs, labels, k):
    groups=sorted([g.item() for g in labels.unique() if g.item()!=-1])
    cent=torch.stack([embs[labels==g].mean(0) for g in groups])
    cent=cent-cent.mean(0,keepdim=True)
    _,_,Vt=torch.linalg.svd(cent,full_matrices=False)
    return Vt.T[:,:min(k,len(groups)-1)]

def project_out(e,B):
    d=e-(e@B)@B.T
    return d/d.norm(dim=-1,keepdim=True).clamp(min=1e-8)

ap=argparse.ArgumentParser()
ap.add_argument("--attribute",required=True)
ap.add_argument("--backbone",default="ViT-B/32")
ap.add_argument("--seed",type=int,default=42)
ap.add_argument("--n_dirs",type=int,default=8)
ap.add_argument("--baseline",action="store_true")
ap.add_argument("--train_cap",type=int,default=5000)
ap.add_argument("--model_path",default=None)
a=ap.parse_args()
cfg=OmegaConf.load("configs/datasets/local_paths.yaml")
dev="cuda" if torch.cuda.is_available() else "cpu"
m=FairCLIP(model_name=a.backbone,device=dev)
if not a.baseline: m.load(a.model_path)
m.eval()
va,val=encode(m,cfg.utkface.manifest,a.attribute,"val",dev,a.backbone,"race_idx_7")
occ=m.backbone.encode_text(OCCUPATION_PROMPTS).cpu().float(); occ=occ/occ.norm(dim=-1,keepdim=True)
if a.baseline:
    img,occ_u,tag=va,occ,"BASELINE"
else:
    tr,trl=encode(m,cfg.fairface.manifest,a.attribute,"train",dev,a.backbone,"race_idx",cap=a.train_cap)
    B=fit_subspace(tr,trl,a.n_dirs)
    img=project_out(va,B); occ_u=project_out(occ,B); tag="FairCLIP"
dpg=compute_dpg_occupation(img,occ_u,val)
eod=compute_eod_occupation(img,occ_u,val)
rbs=compute_rbs(img,val)
print(f"{tag} UTK {a.backbone} {a.attribute} s{a.seed}: DPG={dpg:.4f} EOD={eod:.4f} RBS={rbs:.2e}")
