import torch, numpy as np, csv, os, sys
from omegaconf import OmegaConf
from models.fairclip import FairCLIP
from data.datasets import FaceDataset, collate_dict
from torch.utils.data import DataLoader
from evaluation.metrics.all_metrics import maxskew_ndkl_zhang, ZHANG_NEUTRAL_QUERIES
cfg=OmegaConf.load("configs/datasets/local_paths.yaml")
K={"gender":2,"age":8}; R={"gender":2,"age":3}
PROMPTS={"gender":["a photo of a male person","a photo of a female person"],
 "age":["a photo of a person aged 0-2","a photo of a person aged 3-9","a photo of a person aged 10-19",
        "a photo of a person aged 20-29","a photo of a person aged 30-39","a photo of a person aged 40-49",
        "a photo of a person aged 50-59","a photo of a person aged 60-69","a photo of a person aged 70 or older"]}
def fit(E,l,k):
    g=sorted(l.unique().tolist()); c=torch.stack([E[l==x].mean(0) for x in g])
    _,_,Vt=torch.linalg.svd(c-c.mean(0,keepdim=True),full_matrices=False); return Vt.T[:,:k]
def proj(E,B):
    d=E-(E@B)@B.T; return d/d.norm(dim=-1,keepdim=True).clamp(min=1e-8)
def dpg(pred,l,n):
    return float(np.mean([max([(pred[l==g]==c).float().mean().item() for g in sorted(l.unique().tolist())])
                        - min([(pred[l==g]==c).float().mean().item() for g in sorted(l.unique().tolist())])
                        for c in range(n)]))
rows=[]
for attr in ["gender","age"]:
    for tag, ck in [("Baseline", None), ("FairCLIP", f"results/checkpoints/fairface/ViT-B_32/{attr}/seed42/best_model.pt")]:
        print(f"[{attr}/{tag}] loading model...", flush=True)
        if ck and not os.path.exists(ck): print("  missing ckpt", flush=True); continue
        m=FairCLIP(model_name="ViT-B/32",device="cuda")
        if ck: m.load(ck)
        m.eval()
        ds=FaceDataset(cfg.facet.manifest, split="test", train=False)
        print(f"  dataset size: {len(ds)}", flush=True)
        ld=DataLoader(ds,batch_size=64,shuffle=False,collate_fn=collate_dict,num_workers=2)
        img,labs=[],[]
        with torch.no_grad():
            for i,b in enumerate(ld):
                img.append(m.backbone.encode_images(b["image"].to("cuda")).cpu())
                labs.append(b[attr].clone().detach())
                if i%20==0: print(f"  batch {i}/{len(ld)}", flush=True)
        img=torch.cat(img).float(); labs=torch.cat(labs)
        keep=labs>=0; img,labs=img[keep],labs[keep]
        print(f"  valid labels: {len(labs)}", flush=True)
        if len(labs)==0: print("  NO LABELS - skip", flush=True); continue
        img=img/img.norm(dim=-1,keepdim=True)
        ct=m.encode_text(PROMPTS[attr]).cpu().float(); ct=ct/ct.norm(dim=-1,keepdim=True)
        nt=m.encode_text(ZHANG_NEUTRAL_QUERIES).cpu().float(); nt=nt/nt.norm(dim=-1,keepdim=True)
        pred=(img@ct.T).argmax(1); acc=(pred==labs).float().mean().item()
        d=dpg(pred,labs,len(PROMPTS[attr]))
        ei,et=img.clone(),nt.clone()
        if tag=="FairCLIP":
            for _ in range(R[attr]):
                B=fit(ei,labs,K[attr]); ei=proj(ei,B); et=proj(et,B)
        ms,nd=maxskew_ndkl_zhang(et,ei,labs,k_skew=1000,k_ndkl=1000)
        print(f"RESULT FACET {attr} {tag}: N={len(labs)} Acc={acc:.3f} DPG={d:.3f} MaxSkew={ms:.3f} NDKL={nd:.3f}", flush=True)
        rows.append([attr,tag,len(labs),round(acc,3),round(d,3),round(ms,3),round(nd,3)])
        del m; torch.cuda.empty_cache()
with open("results/tables/facet_crossdataset.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["attribute","method","N","Acc","DPG","MaxSkew","NDKL"]); w.writerows(rows)
print("SAVED results/tables/facet_crossdataset.csv", flush=True)
