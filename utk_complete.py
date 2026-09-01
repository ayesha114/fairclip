import torch, numpy as np, csv, os, sys
from omegaconf import OmegaConf
from models.fairclip import FairCLIP
from data.datasets import FaceDataset, collate_dict
from torch.utils.data import DataLoader
from evaluation.metrics.all_metrics import maxskew_ndkl_zhang, ZHANG_NEUTRAL_QUERIES
cfg=OmegaConf.load("configs/datasets/local_paths.yaml")
OUT="results/UTK_complete_metrics.csv"
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
    g=sorted([x for x in l.unique().tolist() if x!=-1]); c=torch.stack([E[l==x].mean(0) for x in g])
    _,_,Vt=torch.linalg.svd(c-c.mean(0,keepdim=True),full_matrices=False); return Vt.T[:,:k]
def proj(E,B):
    d=E-(E@B)@B.T; return d/d.norm(dim=-1,keepdim=True).clamp(min=1e-8)
def rbs(E,l):
    mu=E.mean(0); return float(np.mean([(E[l==g].mean(0)-mu).norm().item() for g in sorted([x for x in l.unique().tolist() if x!=-1])]))
def eod(pred,l,n):
    v=[]
    for c in range(n):
        t=[];f=[]
        for g in sorted([x for x in l.unique().tolist() if x!=-1]):
            m=l==g; pos=(l[m]==c)
            if pos.sum()>0: t.append((pred[m][pos]==c).float().mean().item())
            if (~pos).sum()>0: f.append((pred[m][~pos]==c).float().mean().item())
        if t: v.append(max(t)-min(t))
        if f: v.append(max(f)-min(f))
    return float(np.mean(v))
def dpg(pred,l,n):
    G=sorted([x for x in l.unique().tolist() if x!=-1])
    return float(np.mean([max([(pred[l==g]==c).float().mean().item() for g in G])-min([(pred[l==g]==c).float().mean().item() for g in G]) for c in range(n)]))
done=set()
if os.path.exists(OUT):
    done={tuple(r[:4]) for r in list(csv.reader(open(OUT)))[1:]}
def add(row):
    new=not os.path.exists(OUT)
    with open(OUT,"a",newline="") as f:
        w=csv.writer(f)
        if new: w.writerow(["backbone","attribute","method","seed","Acc","F1","DPG","EOD","RBS","MaxSkew","NDKL","ABLE"])
        w.writerow(row)
for bb,bbt,bs in [("ViT-B/32","ViT-B_32",64),("ViT-B/16","ViT-B_16",32),("ViT-L/14","ViT-L_14",16),("ViT-H/14","ViT-H_14",8)]:
    for attr in ["gender","age","race"]:
        for method,seeds in [("Baseline",[42]),("FairCLIP",[42,123,456])]:
            for sd in seeds:
                key=(bbt,attr,method,str(sd))
                if key in done: print("SKIP",key,flush=True); continue
                ck=f"results/checkpoints/fairface/{bbt}/{attr}/seed{sd}/best_model.pt"
                if method=="FairCLIP" and not os.path.exists(ck): print("MISS",key,flush=True); continue
                try:
                    m=FairCLIP(model_name=bb,device="cuda")
                    if method=="FairCLIP": m.load(ck)
                    m.eval()
                    ds=FaceDataset(cfg.utkface.manifest,split="val",train=False,race_column="race_idx_7")
                    ld=DataLoader(ds,batch_size=bs,shuffle=False,collate_fn=collate_dict,num_workers=2)
                    img,labs=[],[]
                    with torch.no_grad():
                        for b in ld:
                            img.append(m.backbone.encode_images(b["image"].to("cuda")).cpu()); labs.append(b[attr].clone().detach())
                    img=torch.cat(img).float(); labs=torch.cat(labs); img=img/img.norm(dim=-1,keepdim=True)
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
                    d=dpg(pred,labs,len(P[attr])); e=eod(pred,labs,len(P[attr])); r=rbs(img,labs)
                    ei,et=img.clone(),nt.clone()
                    if method=="FairCLIP":
                        k,rounds=BEST[(bbt,attr)]
                        for _ in range(rounds):
                            B=fit(ei,labs,k); ei=proj(ei,B); et=proj(et,B)
                    ms,nd=maxskew_ndkl_zhang(et,ei,labs,k_skew=1000,k_ndkl=1000)
                    able=2/(1/max(acc,1e-6)+1/np.exp(-ms))*100
                    row=[bbt,attr,method,sd,round(acc,3),round(f1,3),round(d,3),round(e,3),round(r,4),round(ms,3),round(nd,3),round(able,2)]
                    add(row); print("DONE",key,row[4:],flush=True)
                    del m; torch.cuda.empty_cache()
                except Exception as ex:
                    print("FAIL",key,repr(ex)[:150],flush=True); torch.cuda.empty_cache()
print("UTK FINISHED",flush=True)
