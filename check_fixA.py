import torch
from omegaconf import OmegaConf
from models.fairclip import FairCLIP
from data.datasets import FaceDataset, collate_dict
from torch.utils.data import DataLoader
from evaluation.metrics.all_metrics import maxskew_ndkl_zhang, ZHANG_NEUTRAL_QUERIES
cfg=OmegaConf.load("configs/datasets/local_paths.yaml")
m=FairCLIP(model_name="ViT-L/14",device="cuda")
m.load("results/checkpoints_fix/fairface/ViT-L_14/gender/seed42/best_model.pt"); m.eval()
ds=FaceDataset(cfg.fairface.manifest,split="val",train=False)
ld=DataLoader(ds,batch_size=16,shuffle=False,collate_fn=collate_dict,num_workers=2)
img,labs=[],[]
with torch.no_grad():
    for b in ld:
        img.append(m.backbone.encode_images(b["image"].to("cuda")).cpu())
        labs.append(b["gender"].clone().detach())
img=torch.cat(img).float(); labs=torch.cat(labs); img=img/img.norm(dim=-1,keepdim=True)
txt=m.encode_text(ZHANG_NEUTRAL_QUERIES).cpu().float(); txt=txt/txt.norm(dim=-1,keepdim=True)
def fit(E,l,k):
    g=sorted(l.unique().tolist()); c=torch.stack([E[l==x].mean(0) for x in g])
    _,_,Vt=torch.linalg.svd(c-c.mean(0,keepdim=True),full_matrices=False); return Vt.T[:,:k]
def proj(E,B):
    d=E-(E@B)@B.T; return d/d.norm(dim=-1,keepdim=True).clamp(min=1e-8)
print("target: MaxSkew<0.106 AND NDKL<0.035", flush=True)
for k in [1,2]:
    ei,et=img.clone(),txt.clone()
    for r in range(1,13):
        B=fit(ei,labs,k); ei=proj(ei,B); et=proj(et,B)
        ms,nd=maxskew_ndkl_zhang(et,ei,labs,k_skew=1000,k_ndkl=1000)
        flag="BOTH ✓" if (ms<0.106 and nd<0.035) else ("MS" if ms<0.106 else "-")
        print(f"k={k} r={r}: MaxSkew={ms:.4f} NDKL={nd:.4f} {flag}", flush=True)
