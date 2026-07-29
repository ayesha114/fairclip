"""Does projection destroy general meaning, or only remove race?
Test: after each projection round, can embeddings still tell which of
several captions matches an image (a non-demographic task)?
If matching stays reasonable -> projection is clean (real finding).
If matching collapses too -> embeddings are broken (artifact)."""
import torch
from omegaconf import OmegaConf
from models.fairclip import FairCLIP
from data.datasets import FaceDataset, collate_dict
from torch.utils.data import DataLoader
torch.set_num_threads(4)
cfg=OmegaConf.load("configs/datasets/local_paths.yaml")
DEV="cpu"; ATTR="race"
# non-demographic probe captions (about the photo, not the race)
PROBES=["a photo of a person smiling","a photo of a person with short hair",
 "a close-up portrait photo","a photo of a young child","a photo of an elderly person",
 "a black and white photo","a photo of a person wearing glasses","a photo taken outdoors"]
print("loading on CPU...", flush=True)
m=FairCLIP(model_name="ViT-B/32",device=DEV)
m.load(f"results/checkpoints/fairface/ViT-B_32/{ATTR}/seed42/best_model.pt"); m.eval()
ds=FaceDataset(cfg.fairface.manifest,split="val",train=False)
ld=DataLoader(ds,batch_size=32,shuffle=False,collate_fn=collate_dict,num_workers=2)
img,labs=[],[]
with torch.no_grad():
    for b in ld:
        img.append(m.backbone.encode_images(b["image"].to(DEV)).cpu())
        labs.append(b[ATTR].clone().detach())
        if sum(x.shape[0] for x in img)>=1500: break
img=torch.cat(img)[:1500].float(); labs=torch.cat(labs)[:1500]
img=img/img.norm(dim=-1,keepdim=True)
probe=m.encode_text(PROBES).cpu().float(); probe=probe/probe.norm(dim=-1,keepdim=True)
def fit(E,l,k=6):
    g=sorted(l.unique().tolist()); c=torch.stack([E[l==x].mean(0) for x in g])
    _,_,Vt=torch.linalg.svd(c-c.mean(0,keepdim=True),full_matrices=False); return Vt.T[:,:k]
def proj(E,B):
    d=E-(E@B)@B.T; return d/d.norm(dim=-1,keepdim=True).clamp(min=1e-8)
# measure: how spread-out are the probe-caption assignments? (entropy of chosen captions)
# if embeddings alive, different images pick different probes (high spread).
# if broken/collapsed, all images pick the same probe (low spread).
import math
def spread(E):
    picks=(E@probe.T).argmax(1)
    counts=torch.bincount(picks,minlength=len(PROBES)).float()
    p=counts/counts.sum(); p=p[p>0]
    ent=-(p*p.log()).sum().item()/math.log(len(PROBES))  # 0..1, higher=more varied
    return ent, counts.tolist()
print("\nround | caption-variety(0..1) | distribution", flush=True)
ei,eprobe=img.clone(),probe.clone()
e,c=spread(ei); print(f"  0   |   {e:.3f}   | {c}", flush=True)
for r in range(1,9):
    B=fit(ei,labs); ei=proj(ei,B)
    e,c=spread(ei); print(f"  {r}   |   {e:.3f}   | {c}", flush=True)
print("\nIf variety stays >0.5: embeddings keep general meaning (REAL finding).", flush=True)
print("If variety collapses toward 0: embeddings broken (artifact).", flush=True)
