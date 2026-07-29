import torch, csv, numpy as np
from omegaconf import OmegaConf
from models.fairclip import FairCLIP
from data.datasets import FaceDataset, collate_dict
from torch.utils.data import DataLoader
from evaluation.metrics.all_metrics import maxskew_ndkl_zhang, ZHANG_NEUTRAL_QUERIES
cfg = OmegaConf.load("configs/datasets/local_paths.yaml")
def proj(E,B):
    d=E-(E@B)@B.T; return d/d.norm(dim=-1,keepdim=True).clamp(min=1e-8)
rows=[]
for bb,bbt in [("ViT-B/16","ViT-B_16"),("ViT-L/14","ViT-L_14")]:
    for attr in ["gender","age","race"]:
        ck=f"results/checkpoints/fairface/{bbt}/{attr}/seed42/best_model.pt"
        import os
        if not os.path.exists(ck): print(f"missing {ck}"); continue
        m=FairCLIP(model_name=bb,device="cuda"); m.load(ck); m.eval()
        ds=FaceDataset(cfg.fairface.manifest,split="val",train=False)
        ld=DataLoader(ds,batch_size=32,shuffle=False,collate_fn=collate_dict,num_workers=2)
        raw,labs=[],[]
        with torch.no_grad():
            for b in ld:
                raw.append(m.backbone.encode_images(b["image"].to("cuda")).cpu())
                labs.append(b[attr].clone().detach())
        raw=torch.cat(raw).float(); labs=torch.cat(labs)
        raw=raw/raw.norm(dim=-1,keepdim=True)
        txt=m.encode_text(ZHANG_NEUTRAL_QUERIES).cpu().float(); txt=txt/txt.norm(dim=-1,keepdim=True)
        groups=sorted(labs.unique().tolist())
        cents=torch.stack([raw[labs==g].mean(0) for g in groups])
        diffs=cents-cents.mean(0,keepdim=True)
        _,_,Vt=torch.linalg.svd(diffs,full_matrices=False)
        Q,_=torch.linalg.qr(torch.cat([Vt.T,m.bias_remover.image_bias_directions.cpu().float()],dim=1))
        B=Q[:,:min(Q.shape[1],Vt.T.shape[1]+5)]
        ms,nd=maxskew_ndkl_zhang(proj(txt,B),proj(raw,B),labs,k_skew=1000,k_ndkl=1000)
        print(f"{bbt} {attr}: MaxSkew={ms:.3f} NDKL={nd:.3f}")
        rows.append([bbt,attr,round(ms,3),round(nd,3)])
        del m; torch.cuda.empty_cache()
with open("results/tables/maxskew_backbones.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["backbone","attribute","MaxSkew","NDKL"]); w.writerows(rows)
print("saved results/tables/maxskew_backbones.csv")
