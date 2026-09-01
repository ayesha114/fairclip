import torch, numpy as np, csv, os
from omegaconf import OmegaConf
from models.fairclip import FairCLIP
from data.datasets import FaceDataset, collate_dict
from torch.utils.data import DataLoader
from evaluation.metrics.all_metrics import maxskew_ndkl_zhang, ZHANG_NEUTRAL_QUERIES
cfg=OmegaConf.load("configs/datasets/local_paths.yaml")
OUT="results/ABLATION_FULL_r1.csv"
BEST={"gender":(2,1),"age":(8,1),"race":(6,1)}   # B/32 settings
P={"gender":["a photo of a male person","a photo of a female person"],
 "age":["a photo of a person aged 0-2","a photo of a person aged 3-9","a photo of a person aged 10-19",
  "a photo of a person aged 20-29","a photo of a person aged 30-39","a photo of a person aged 40-49",
  "a photo of a person aged 50-59","a photo of a person aged 60-69","a photo of a person aged 70 or older"],
 "race":["a photo of a White person","a photo of a Black person","a photo of a Latino Hispanic person",
  "a photo of an East Asian person","a photo of a Southeast Asian person","a photo of an Indian person",
  "a photo of a Middle Eastern person"]}
def fit(E,l,k):
    g=sorted([x for x in l.unique().tolist() if x!=-1]); c=torch.stack([E[l==x].mean(0) for x in g])
    _,_,Vt=torch.linalg.svd(c-c.mean(0,keepdim=True),full_matrices=False); return Vt.T[:,:k]
def proj(E,B):
    d=E-(E@B)@B.T; return d/d.norm(dim=-1,keepdim=True).clamp(min=1e-8)
done=set()
if os.path.exists(OUT): done={tuple(r[:2]) for r in list(csv.reader(open(OUT)))[1:]}
def add(r):
    n=not os.path.exists(OUT)
    with open(OUT,"a",newline="") as f:
        w=csv.writer(f)
        if n: w.writerow(["config","attribute","Acc","F1","MaxSkew","NDKL","ABLE"])
        w.writerow(r)
CFGS=[("FULL","results/checkpoints_procfix/fairface/ViT-B_32/{a}/seed42/best_model.pt")]+\
     [(n,"results/checkpoints_ablfix_%s/fairface/ViT-B_32/{a}/seed42/best_model.pt"%n)
      for n in ["no_fairloss","no_temp","no_trainproj","no_procrustes","no_subspace"]]
for name,tpl in CFGS:
    for attr in ["gender","age","race"]:
        key=(name,attr)
        if key in done: print("SKIP",key,flush=True); continue
        ck=tpl.format(a=attr)
        if not os.path.exists(ck): print("MISS",key,flush=True); continue
        try:
            m=FairCLIP(model_name="ViT-B/32",device="cuda"); m.load(ck); m.eval()
            ds=FaceDataset(cfg.fairface.manifest,split="val",train=False)
            ld=DataLoader(ds,batch_size=64,shuffle=False,collate_fn=collate_dict,num_workers=2)
            img,labs=[],[]
            with torch.no_grad():
                for b in ld:
                    img.append(m.backbone.encode_images(b["image"].to("cuda")).cpu())
                    labs.append(b[attr].clone().detach())
            img=torch.cat(img).float(); labs=torch.cat(labs)
            keep=labs>=0; img,labs=img[keep],labs[keep]
            img=img/img.norm(dim=-1,keepdim=True)
            ct=m.encode_text(P[attr]).cpu().float(); ct=ct/ct.norm(dim=-1,keepdim=True)
            nt=m.encode_text(ZHANG_NEUTRAL_QUERIES).cpu().float(); nt=nt/nt.norm(dim=-1,keepdim=True)
            pred=(img@ct.T).argmax(1); acc=(pred==labs).float().mean().item()
            f1s=[]
            for c in labs.unique():
                p=(pred==c); g=(labs==c)
                tp=(p&g).sum().item(); fp=(p&(~g)).sum().item(); fn=((~p)&g).sum().item()
                pr=tp/(tp+fp) if tp+fp else 0.0; rc=tp/(tp+fn) if tp+fn else 0.0
                f1s.append(2*pr*rc/(pr+rc) if pr+rc else 0.0)
            f1=float(np.mean(f1s))
            ei,et=img.clone(),nt.clone()
            k,rounds=BEST[attr]
            for _ in range(rounds):
                B=fit(ei,labs,k); ei=proj(ei,B); et=proj(et,B)
            ms,nd=maxskew_ndkl_zhang(et,ei,labs,k_skew=1000,k_ndkl=1000)
            able=2/(1/max(acc,1e-6)+1/np.exp(-ms))*100
            row=[name,attr,round(acc,3),round(f1,3),round(ms,3),round(nd,3),round(able,2)]
            add(row); print("DONE",key,row[2:],flush=True)
            del m; torch.cuda.empty_cache()
        except Exception as e:
            print("FAIL",key,repr(e)[:150],flush=True); torch.cuda.empty_cache()
print("ABLATION FULL FINISHED",flush=True)
