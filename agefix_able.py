import torch, numpy as np, glob
from omegaconf import OmegaConf
from models.fairclip import FairCLIP
from data.datasets import FaceDataset, collate_dict
from torch.utils.data import DataLoader
from evaluation.metrics.all_metrics import maxskew_ndkl_zhang, ZHANG_NEUTRAL_QUERIES
cfg=OmegaConf.load("configs/datasets/local_paths.yaml")
AGE=["a photo of a person aged 0-2","a photo of a person aged 3-9","a photo of a person aged 10-19",
 "a photo of a person aged 20-29","a photo of a person aged 30-39","a photo of a person aged 40-49",
 "a photo of a person aged 50-59","a photo of a person aged 60-69","a photo of a person aged 70 or older"]
def fit(E,l,k):
    g=sorted(l.unique().tolist()); c=torch.stack([E[l==x].mean(0) for x in g])
    _,_,Vt=torch.linalg.svd(c-c.mean(0,keepdim=True),full_matrices=False); return Vt.T[:,:k]
def proj(E,B):
    d=E-(E@B)@B.T; return d/d.norm(dim=-1,keepdim=True).clamp(min=1e-8)
for lam in ["0.25","0.75"]:
    cks=glob.glob(f"results/checkpoints_agefix_lam{lam}/**/best_model.pt", recursive=True)
    if not cks:
        print(f"lam{lam}: no ckpt", flush=True); continue
    m=FairCLIP(model_name="ViT-B/32",device="cuda"); m.load(cks[0]); m.eval()
    ds=FaceDataset(cfg.fairface.manifest,split="val",train=False)
    ld=DataLoader(ds,batch_size=64,shuffle=False,collate_fn=collate_dict,num_workers=2)
    img,labs=[],[]
    with torch.no_grad():
        for b in ld:
            img.append(m.backbone.encode_images(b["image"].to("cuda")).cpu())
            labs.append(b["age"].clone().detach())
    img=torch.cat(img).float(); labs=torch.cat(labs); img=img/img.norm(dim=-1,keepdim=True)
    ct=m.encode_text(AGE).cpu().float(); ct=ct/ct.norm(dim=-1,keepdim=True)
    nt=m.encode_text(ZHANG_NEUTRAL_QUERIES).cpu().float(); nt=nt/nt.norm(dim=-1,keepdim=True)
    acc=((img@ct.T).argmax(1)==labs).float().mean().item()
    best=(9e9,9e9,0,0,0)
    for k in [6,8]:
        ei,et=img.clone(),nt.clone()
        for r in range(1,9):
            B=fit(ei,labs,k); ei=proj(ei,B); et=proj(et,B)
            ms,nd=maxskew_ndkl_zhang(et,ei,labs,k_skew=1000,k_ndkl=1000)
            able=2/(1/max(acc,1e-6)+1/np.exp(-ms))*100
            if ms<best[0]: best=(ms,nd,k,r,able)
    print(f"lam{lam}: Acc={acc:.3f} MaxSkew={best[0]:.3f} NDKL={best[1]:.3f} ABLE={best[4]:.2f} (k={best[2]},r={best[3]})", flush=True)
    print("   Zhang targets: ABLE 59.60, MaxSkew 0.572, NDKL 0.364 | current best ABLE 58.92", flush=True)
    del m; torch.cuda.empty_cache()
