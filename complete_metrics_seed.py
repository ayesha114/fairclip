import torch, numpy as np, csv, os, sys
SEED=sys.argv[1]
from omegaconf import OmegaConf
from models.fairclip import FairCLIP
from data.datasets import FaceDataset, collate_dict
from torch.utils.data import DataLoader
from evaluation.metrics.all_metrics import maxskew_ndkl_zhang, ZHANG_NEUTRAL_QUERIES
cfg=OmegaConf.load("configs/datasets/local_paths.yaml")
BEST={("ViT-B_32","gender"):(2,2),("ViT-B_32","age"):(8,3),("ViT-B_32","race"):(6,5),
      ("ViT-B_16","gender"):(1,8),("ViT-B_16","age"):(6,8),("ViT-B_16","race"):(4,5),
      ("ViT-L_14","gender"):(1,8),("ViT-L_14","age"):(6,7),("ViT-L_14","race"):(6,4),
      ("ViT-H_14","gender"):(2,4),("ViT-H_14","age"):(6,5),("ViT-H_14","race"):(6,5)}
P={"gender":["a photo of a male person","a photo of a female person"],
 "age":["a photo of a person aged 0-2","a photo of a person aged 3-9","a photo of a person aged 10-19",
  "a photo of a person aged 20-29","a photo of a person aged 30-39","a photo of a person aged 40-49",
  "a photo of a person aged 50-59","a photo of a person aged 60-69","a photo of a person aged 70 or older"],
 "race":["a photo of a White person","a photo of a Black person","a photo of a Latino Hispanic person",
  "a photo of an East Asian person","a photo of a Southeast Asian person","a photo of an Indian person",
  "a photo of a Middle Eastern person"]}
def fit(E,l,k):
    g=sorted(l.unique().tolist()); c=torch.stack([E[l==x].mean(0) for x in g])
    _,_,Vt=torch.linalg.svd(c-c.mean(0,keepdim=True),full_matrices=False); return Vt.T[:,:k]
def proj(E,B):
    d=E-(E@B)@B.T; return d/d.norm(dim=-1,keepdim=True).clamp(min=1e-8)
def rbs(E,l):
    mu=E.mean(0); return float(np.mean([(E[l==g].mean(0)-mu).norm().item() for g in sorted(l.unique().tolist())]))
def eod(pred,l,n):
    v=[]
    for c in range(n):
        t=[];f=[]
        for g in sorted(l.unique().tolist()):
            m=l==g; pos=(l[m]==c)
            if pos.sum()>0: t.append((pred[m][pos]==c).float().mean().item())
            if (~pos).sum()>0: f.append((pred[m][~pos]==c).float().mean().item())
        if t: v.append(max(t)-min(t))
        if f: v.append(max(f)-min(f))
    return float(np.mean(v))
def dpg(pred,l,n):
    return float(np.mean([max([(pred[l==g]==c).float().mean().item() for g in sorted(l.unique().tolist())])
                        - min([(pred[l==g]==c).float().mean().item() for g in sorted(l.unique().tolist())]) for c in range(n)]))
rows=[]
for bb,bbt,bs in [("ViT-B/16","ViT-B_16",32),("ViT-L/14","ViT-L_14",16),("ViT-H/14","ViT-H_14",8)]:
    for attr in ["gender","age","race"]:
        for method in ["FairCLIP"]:
            ck=f"results/checkpoints/fairface/{bbt}/{attr}/seed{SEED}/best_model.pt"
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
            ct=m.encode_text(P[attr]).cpu().float(); ct=ct/ct.norm(dim=-1,keepdim=True)
            nt=m.encode_text(ZHANG_NEUTRAL_QUERIES).cpu().float(); nt=nt/nt.norm(dim=-1,keepdim=True)
            pred=(img@ct.T).argmax(1); acc=(pred==labs).float().mean().item()
            f1=acc  # macro-F1 approximated by eval_all; keep acc-based ABLE
            d=dpg(pred,labs,len(P[attr])); e=eod(pred,labs,len(P[attr])); r=rbs(img,labs)
            ei,et=img.clone(),nt.clone()
            if method=="FairCLIP":
                k,rounds=BEST[(bbt,attr)]
                for _ in range(rounds):
                    B=fit(ei,labs,k); ei=proj(ei,B); et=proj(et,B)
            ms,nd=maxskew_ndkl_zhang(et,ei,labs,k_skew=1000,k_ndkl=1000)
            able=2/(1/max(acc,1e-6)+1/np.exp(-ms))*100
            print(f"{bbt} {attr} {method}: Acc={acc:.3f} DPG={d:.3f} EOD={e:.3f} RBS={r:.4f} MaxSkew={ms:.3f} NDKL={nd:.3f} ABLE={able:.2f}", flush=True)
            rows.append([bbt,attr,method,round(acc,3),round(d,3),round(e,3),round(r,4),round(ms,3),round(nd,3),round(able,2)])
            del m; torch.cuda.empty_cache()
with open(f"results/tables/complete_metrics_seed{SEED}.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["backbone","attribute","method","Acc","DPG","EOD","RBS","MaxSkew","NDKL","ABLE"]); w.writerows(rows)
print("saved results/tables/complete_metrics_seed{SEED}.csv", flush=True)
