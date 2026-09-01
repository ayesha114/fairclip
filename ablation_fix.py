"""Ablation, same protocol as eval_methodology_correct.py: fit subspace on TRAIN, apply once."""
import torch, numpy as np, csv, os
from omegaconf import OmegaConf
from models.fairclip import FairCLIP
from data.datasets import FaceDataset, collate_dict
from torch.utils.data import DataLoader
from evaluation.metrics.all_metrics import (maxskew_ndkl_zhang, ZHANG_NEUTRAL_QUERIES,
    compute_dpg_occupation, compute_eod_occupation, compute_rbs, OCCUPATION_PROMPTS)
cfg=OmegaConf.load("configs/datasets/local_paths.yaml")
OUT="results/ABLATION_FIXED.csv"; NDIRS=8; CAP=20000
P={"gender":["a photo of a male person","a photo of a female person"],
 "age":["a photo of a person aged 0-2","a photo of a person aged 3-9","a photo of a person aged 10-19",
  "a photo of a person aged 20-29","a photo of a person aged 30-39","a photo of a person aged 40-49",
  "a photo of a person aged 50-59","a photo of a person aged 60-69","a photo of a person aged 70 or older"],
 "race":["a photo of a White person","a photo of a Black person","a photo of a Latino Hispanic person",
  "a photo of an East Asian person","a photo of a Southeast Asian person","a photo of an Indian person",
  "a photo of a Middle Eastern person"]}
def enc(m,split,attr,cap=None):
    ds=FaceDataset(cfg.fairface.manifest,split=split,train=False)
    ld=DataLoader(ds,batch_size=64,shuffle=False,collate_fn=collate_dict,num_workers=2)
    E,L=[],[]
    with torch.no_grad():
        for b in ld:
            E.append(m.backbone.encode_images(b["image"].to("cuda")).cpu()); L.append(b[attr].clone().detach())
            if cap and sum(x.shape[0] for x in E)>=cap: break
    E=torch.cat(E).float(); L=torch.cat(L)
    if cap: E,L=E[:cap],L[:cap]
    return E/E.norm(dim=-1,keepdim=True),L
def fit(E,l,k):
    g=sorted([x for x in l.unique().tolist() if x!=-1]); c=torch.stack([E[l==x].mean(0) for x in g])
    c=c-c.mean(0,keepdim=True); _,_,Vt=torch.linalg.svd(c,full_matrices=False)
    return Vt.T[:,:min(k,len(g)-1)]
def prj(E,B):
    d=E-(E@B)@B.T; return d/d.norm(dim=-1,keepdim=True).clamp(min=1e-8)
done=set()
if os.path.exists(OUT): done={tuple(r[:2]) for r in list(csv.reader(open(OUT)))[1:]}
def add(r):
    n=not os.path.exists(OUT)
    with open(OUT,"a",newline="") as f:
        w=csv.writer(f)
        if n: w.writerow(["config","attribute","Acc","F1","DPG","EOD","RBS","MaxSkew","NDKL","ABLE"])
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
            va,val=enc(m,"val",attr); tr,trl=enc(m,"train",attr,CAP)
            B=fit(tr,trl,NDIRS); del tr,trl
            occ=m.backbone.encode_text(OCCUPATION_PROMPTS).cpu().float(); occ=occ/occ.norm(dim=-1,keepdim=True)
            ct=m.encode_text(P[attr]).cpu().float(); ct=ct/ct.norm(dim=-1,keepdim=True)
            nt=m.encode_text(ZHANG_NEUTRAL_QUERIES).cpu().float(); nt=nt/nt.norm(dim=-1,keepdim=True)
            img=prj(va,B); occ_u=prj(occ,B); ct_u=prj(ct,B); nt_u=prj(nt,B)
            d=compute_dpg_occupation(img,occ_u,val); e=compute_eod_occupation(img,occ_u,val); r_=compute_rbs(img,val)
            v=val!=-1; pred=(va[v].float()@ct.float().T).argmax(1); gt=val[v]   # Acc/F1 on un-projected (utility)
            acc=float((pred==gt).float().mean().item())
            f1s=[]
            for c in gt.unique():
                p=(pred==c); g=(gt==c)
                tp=(p&g).sum().item(); fp=(p&(~g)).sum().item(); fn=((~p)&g).sum().item()
                pr=tp/(tp+fp) if tp+fp else 0.0; rc=tp/(tp+fn) if tp+fn else 0.0
                f1s.append(2*pr*rc/(pr+rc) if pr+rc else 0.0)
            f1=float(np.mean(f1s))
            ms,nd=maxskew_ndkl_zhang(nt_u,img,val,k_skew=1000,k_ndkl=1000)
            able=2/(1/max(acc,1e-6)+1/np.exp(-ms))*100
            row=[name,attr,round(acc,4),round(f1,4),round(d,4),round(e,4),"%.2e"%r_,round(ms,3),round(nd,3),round(able,2)]
            add(row); print("DONE",key,row[2:],flush=True)
            del m; torch.cuda.empty_cache()
        except Exception as ex:
            print("FAIL",key,repr(ex)[:150],flush=True); torch.cuda.empty_cache()
print("ABLATION FIXED FINISHED",flush=True)
