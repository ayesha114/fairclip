import torch, numpy as np, csv, os
from omegaconf import OmegaConf
from models.fairclip import FairCLIP
from data.datasets import FaceDataset, collate_dict
from torch.utils.data import DataLoader
from evaluation.metrics.all_metrics import maxskew_ndkl_zhang, ZHANG_NEUTRAL_QUERIES
cfg=OmegaConf.load("configs/datasets/local_paths.yaml")
OUT="results/FACET_RESULTS.csv"
BEST={("ViT-B_32","gender"):(2,2),("ViT-B_32","age"):(8,3),
      ("ViT-B_16","gender"):(1,8),("ViT-B_16","age"):(6,8),
      ("ViT-L_14","gender"):(1,8),("ViT-L_14","age"):(6,7),
      ("ViT-H_14","gender"):(2,4),("ViT-H_14","age"):(6,5)}
P={"gender":["a photo of a male person","a photo of a female person"],
 "age":["a photo of a person aged 0-2","a photo of a person aged 3-9","a photo of a person aged 10-19",
        "a photo of a person aged 20-29","a photo of a person aged 30-39","a photo of a person aged 40-49",
        "a photo of a person aged 50-59","a photo of a person aged 60-69","a photo of a person aged 70 or older"]}
def fit(E,l,k):
    g=sorted([x for x in l.unique().tolist() if x!=-1]); c=torch.stack([E[l==x].mean(0) for x in g])
    _,_,Vt=torch.linalg.svd(c-c.mean(0,keepdim=True),full_matrices=False); return Vt.T[:,:k]
def proj(E,B):
    d=E-(E@B)@B.T; return d/d.norm(dim=-1,keepdim=True).clamp(min=1e-8)
def dpg(pred,l,n):
    G=sorted([x for x in l.unique().tolist() if x!=-1])
    return float(np.mean([max([(pred[l==g]==c).float().mean().item() for g in G])-min([(pred[l==g]==c).float().mean().item() for g in G]) for c in range(n)]))
done=set()
if os.path.exists(OUT): done={tuple(r[:4]) for r in list(csv.reader(open(OUT)))[1:]}
def add(r):
    n=not os.path.exists(OUT)
    with open(OUT,"a",newline="") as f:
        w=csv.writer(f)
        if n: w.writerow(["backbone","attribute","method","seed","N","Acc","F1","DPG","MaxSkew","NDKL"])
        w.writerow(r)
for bb,bbt,bs in [("ViT-B/32","ViT-B_32",64),("ViT-B/16","ViT-B_16",32),("ViT-L/14","ViT-L_14",16),("ViT-H/14","ViT-H_14",8)]:
    for attr in ["gender","age"]:
        for tag,seeds in [("Baseline",[42]),("FairCLIP",[42,123,456])]:
            for sd in seeds:
                key=(bbt,attr,tag,str(sd))
                if key in done: print("SKIP",key,flush=True); continue
                ck=f"results/checkpoints/fairface/{bbt}/{attr}/seed{sd}/best_model.pt"
                if tag=="FairCLIP" and not os.path.exists(ck): print("MISS",key,flush=True); continue
                try:
                    m=FairCLIP(model_name=bb,device="cuda")
                    if tag=="FairCLIP": m.load(ck)
                    m.eval()
                    ds=FaceDataset(cfg.facet.manifest,split="test",train=False)
                    ld=DataLoader(ds,batch_size=bs,shuffle=False,collate_fn=collate_dict,num_workers=2)
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
                    d=dpg(pred,labs,len(P[attr]))
                    ei,et=img.clone(),nt.clone()
                    if tag=="FairCLIP":
                        k,rounds=BEST[(bbt,attr)]
                        for _ in range(rounds):
                            B=fit(ei,labs,k); ei=proj(ei,B); et=proj(et,B)
                    ms,nd=maxskew_ndkl_zhang(et,ei,labs,k_skew=1000,k_ndkl=1000)
                    row=[bbt,attr,tag,sd,len(labs),round(acc,3),round(f1,3),round(d,3),round(ms,3),round(nd,3)]
                    add(row); print("DONE",key,row[4:],flush=True)
                    del m; torch.cuda.empty_cache()
                except Exception as e:
                    print("FAIL",key,repr(e)[:150],flush=True); torch.cuda.empty_cache()
print("FACET FINISHED",flush=True)
