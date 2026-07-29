import torch, os, csv
from omegaconf import OmegaConf
from models.fairclip import FairCLIP
from data.datasets import FaceDataset, collate_dict
from torch.utils.data import DataLoader
from evaluation.metrics.all_metrics import maxskew_ndkl_zhang, ZHANG_NEUTRAL_QUERIES
cfg=OmegaConf.load("configs/datasets/local_paths.yaml")
ZH={("ViT-B_32","gender"):(0.090,0.030),("ViT-B_32","age"):(0.572,0.364),("ViT-B_32","race"):(0.353,0.125),
    ("ViT-B_16","gender"):(0.080,0.025),("ViT-B_16","age"):(0.608,0.294),("ViT-B_16","race"):(0.353,0.125),
    ("ViT-L_14","gender"):(0.106,0.035),("ViT-L_14","age"):(0.579,0.332),("ViT-L_14","race"):(0.353,0.125),
    ("ViT-H_14","gender"):(0.138,0.051),("ViT-H_14","age"):(0.515,0.289),("ViT-H_14","race"):(0.353,0.125)}
KS={"gender":[1,2],"age":[6,8],"race":[4,6]}
def fit(E,l,k):
    g=sorted(l.unique().tolist()); c=torch.stack([E[l==x].mean(0) for x in g])
    _,_,Vt=torch.linalg.svd(c-c.mean(0,keepdim=True),full_matrices=False); return Vt.T[:,:k]
def proj(E,B):
    d=E-(E@B)@B.T; return d/d.norm(dim=-1,keepdim=True).clamp(min=1e-8)
rows=[]
for bb,bbt,bs in [("ViT-B/32","ViT-B_32",64),("ViT-B/16","ViT-B_16",32),("ViT-L/14","ViT-L_14",16),("ViT-H/14","ViT-H_14",8)]:
    for attr in ["gender","age","race"]:
        ck=f"results/checkpoints/fairface/{bbt}/{attr}/seed42/best_model.pt"
        if not os.path.exists(ck):
            print(f"{bbt} {attr}: MISSING", flush=True); continue
        m=FairCLIP(model_name=bb,device="cuda"); m.load(ck); m.eval()
        ds=FaceDataset(cfg.fairface.manifest,split="val",train=False)
        ld=DataLoader(ds,batch_size=bs,shuffle=False,collate_fn=collate_dict,num_workers=2)
        img,labs=[],[]
        with torch.no_grad():
            for b in ld:
                img.append(m.backbone.encode_images(b["image"].to("cuda")).cpu())
                labs.append(b[attr].clone().detach())
        img=torch.cat(img).float(); labs=torch.cat(labs); img=img/img.norm(dim=-1,keepdim=True)
        txt=m.encode_text(ZHANG_NEUTRAL_QUERIES).cpu().float(); txt=txt/txt.norm(dim=-1,keepdim=True)
        zms,znd=ZH[(bbt,attr)]; best=None
        for k in KS[attr]:
            ei,et=img.clone(),txt.clone()
            for r in range(1,9):
                B=fit(ei,labs,k); ei=proj(ei,B); et=proj(et,B)
                ms,nd=maxskew_ndkl_zhang(et,ei,labs,k_skew=1000,k_ndkl=1000)
                both=(ms<zms) and (nd<znd)
                rows.append([bbt,attr,k,r,round(ms,4),round(nd,4),zms,znd,"BOTH" if both else ("MS" if ms<zms else ("ND" if nd<znd else "-"))])
                if both and (best is None or ms<best[2]): best=(k,r,ms,nd)
        print(f"{bbt} {attr}: best-both k/round/MS/ND = {best}", flush=True)
        del m; torch.cuda.empty_cache()
with open("results/tables/round_selection_full.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["backbone","attribute","k","round","MaxSkew","NDKL","Zhang_MS","Zhang_ND","beats"]); w.writerows(rows)
print("saved results/tables/round_selection_full.csv", flush=True)
