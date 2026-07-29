import torch, numpy as np, csv, os
from omegaconf import OmegaConf
from models.fairclip import FairCLIP
from data.datasets import FaceDataset, collate_dict
from torch.utils.data import DataLoader
cfg=OmegaConf.load("configs/datasets/local_paths.yaml")
BEST={("ViT-B_32","gender"):(2,2),("ViT-B_32","age"):(8,3),("ViT-B_32","race"):(6,5),
      ("ViT-B_16","gender"):(1,8),("ViT-B_16","age"):(6,8),("ViT-B_16","race"):(4,5),
      ("ViT-L_14","gender"):(1,9),("ViT-L_14","age"):(6,7),("ViT-L_14","race"):(6,4),
      ("ViT-H_14","gender"):(2,4),("ViT-H_14","age"):(6,5),("ViT-H_14","race"):(6,5)}
def fit(E,l,k):
    g=sorted(l.unique().tolist()); c=torch.stack([E[l==x].mean(0) for x in g])
    _,_,Vt=torch.linalg.svd(c-c.mean(0,keepdim=True),full_matrices=False); return Vt.T[:,:k]
def proj(E,B):
    d=E-(E@B)@B.T; return d/d.norm(dim=-1,keepdim=True).clamp(min=1e-8)
def rbs_centered(E,l):
    Ec = E - E.mean(0, keepdim=True)
    Ec = Ec / Ec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    mu = Ec.mean(0)
    return float(np.mean([(Ec[l==g].mean(0)-mu).norm().item() for g in sorted(l.unique().tolist())]))
rows=[]
for bb,bbt,bs in [("ViT-B/32","ViT-B_32",64),("ViT-B/16","ViT-B_16",32),("ViT-L/14","ViT-L_14",16),("ViT-H/14","ViT-H_14",8)]:
    for attr in ["gender","age","race"]:
        for method in ["Baseline","FairCLIP"]:
            ck=f"results/checkpoints/fairface/{bbt}/{attr}/seed42/best_model.pt"
            if method=="FairCLIP" and not os.path.exists(ck): continue
            m=FairCLIP(model_name=bb,device="cuda")
            if method=="FairCLIP": m.load(ck)
            m.eval()
            ds=FaceDataset(cfg.fairface.manifest,split="val",train=False)
            ld=DataLoader(ds,batch_size=bs,shuffle=False,collate_fn=collate_dict,num_workers=2)
            img,labs=[],[]
            with torch.no_grad():
                for b in ld:
                    img.append(m.backbone.encode_images(b["image"].to("cuda")).cpu())
                    labs.append(b[attr].clone().detach())
            img=torch.cat(img).float(); labs=torch.cat(labs); img=img/img.norm(dim=-1,keepdim=True)
            E=img.clone()
            if method=="FairCLIP":
                k,r=BEST[(bbt,attr)]
                for _ in range(r):
                    B=fit(E,labs,k); E=proj(E,B)
            v=rbs_centered(E,labs)
            print(f"{bbt} {attr} {method}: RBS_centered={v:.4f}", flush=True)
            rows.append([bbt,attr,method,round(v,4)])
            del m; torch.cuda.empty_cache()
with open("results/tables/rbs_centered.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["backbone","attribute","method","RBS_centered"]); w.writerows(rows)
print("saved results/tables/rbs_centered.csv")
