#!/usr/bin/env bash
cd /home/ayeshakalsoom/Downloads/thesis
source /home/ayeshakalsoom/Downloads/fairclip-thesis/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while [ ! -f h14_race_final_done.txt ]; do sleep 300; done
echo "[gaps] starting"

# GAP 1: ABLE / EOD / RBS under final protocol (B/32, all attrs, 3 seeds)
python3 - << 'PY' > gap1_able_eod_rbs.txt 2>&1
import torch, numpy as np, csv
from omegaconf import OmegaConf
from models.fairclip import FairCLIP
from data.datasets import FaceDataset, collate_dict
from torch.utils.data import DataLoader
from evaluation.metrics.all_metrics import maxskew_ndkl_zhang, ZHANG_NEUTRAL_QUERIES
cfg=OmegaConf.load("configs/datasets/local_paths.yaml")
K={"gender":2,"age":8,"race":6}; R={"gender":2,"age":3,"race":5}
PROMPTS={"gender":["a photo of a male person","a photo of a female person"],
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
    mu=E.mean(0); return float(np.mean([ (E[l==g].mean(0)-mu).norm().item() for g in sorted(l.unique().tolist())]))
def eod(pred,l,ncls):
    vals=[]
    for c in range(ncls):
        tpr=[]; fpr=[]
        for g in sorted(l.unique().tolist()):
            m=l==g; pos=(l[m]==c); neg=~pos
            if pos.sum()>0: tpr.append((pred[m][pos]==c).float().mean().item())
            if neg.sum()>0: fpr.append((pred[m][neg]==c).float().mean().item())
        if tpr: vals.append(max(tpr)-min(tpr))
        if fpr: vals.append(max(fpr)-min(fpr))
    return float(np.mean(vals))
rows=[]
for attr in ["gender","age","race"]:
    for seed in [42,123,456]:
        import os
        ck=f"results/checkpoints/fairface/ViT-B_32/{attr}/seed{seed}/best_model.pt"
        if not os.path.exists(ck): continue
        m=FairCLIP(model_name="ViT-B/32",device="cuda"); m.load(ck); m.eval()
        ds=FaceDataset(cfg.fairface.manifest,split="val",train=False)
        ld=DataLoader(ds,batch_size=64,shuffle=False,collate_fn=collate_dict,num_workers=2)
        img,labs=[],[]
        with torch.no_grad():
            for b in ld:
                img.append(m.backbone.encode_images(b["image"].to("cuda")).cpu())
                labs.append(b[attr].clone().detach())
        img=torch.cat(img).float(); labs=torch.cat(labs); img=img/img.norm(dim=-1,keepdim=True)
        ct=m.encode_text(PROMPTS[attr]).cpu().float(); ct=ct/ct.norm(dim=-1,keepdim=True)
        nt=m.encode_text(ZHANG_NEUTRAL_QUERIES).cpu().float(); nt=nt/nt.norm(dim=-1,keepdim=True)
        # classification protocol (1-round, as trained)
        pred=(img@ct.T).argmax(1); acc=(pred==labs).float().mean().item()
        e=eod(pred,labs,len(PROMPTS[attr])); r=rbs(img,labs)
        # retrieval protocol (iterative) -> MaxSkew -> ABLE
        ei,et=img.clone(),nt.clone()
        for _ in range(R[attr]):
            B=fit(ei,labs,K[attr]); ei=proj(ei,B); et=proj(et,B)
        ms,nd=maxskew_ndkl_zhang(et,ei,labs,k_skew=1000,k_ndkl=1000)
        able=2/(1/max(acc,1e-6)+1/np.exp(-ms))*100
        print(f"{attr} seed{seed}: ABLE={able:.2f} EOD={e:.3f} RBS={r:.4f} MaxSkew={ms:.3f}")
        rows.append([attr,seed,round(able,2),round(e,3),round(r,4),round(ms,3),round(nd,3)])
        del m; torch.cuda.empty_cache()
with open("results/tables/gap1_able_eod_rbs.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["attribute","seed","ABLE","EOD","RBS","MaxSkew","NDKL"]); w.writerows(rows)
PY

# GAP 2: no_subspace ablation (FP32 to avoid FP16 scaler crash)
python3 - << 'PY'
f="training/train.py"; s=open(f).read()
if "_USE_AMP = not " not in s:
    s=s.replace("_USE_AMP", "_USE_AMP", 1)
open(f,"w").write(s)
PY
python -m training.train --dataset fairface --attribute race --backbone ViT-B/32 \
  --epochs 30 --patience 3 --batch_size 64 --lambda_fair 2.0 --n_bias_dirs 5 --seed 42 \
  --no_subspace --no_train_proj --output_dir results/checkpoints_abl_no_subspace2 > abl_no_subspace2.txt 2>&1
CK=$(find results/checkpoints_abl_no_subspace2 -name best_model.pt | head -1)
[ -n "$CK" ] && python -m evaluation.eval_all --dataset fairface --model_path "$CK" \
  --backbone ViT-B/32 --attribute race > abl_eval_no_subspace2.txt 2>&1

# GAP 3: centroid movement analysis + Grad-CAM
python3 - << 'PY' > gap3_centroid_gradcam.txt 2>&1
import torch, numpy as np, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from models.fairclip import FairCLIP
from data.datasets import FaceDataset, collate_dict
from torch.utils.data import DataLoader
cfg=OmegaConf.load("configs/datasets/local_paths.yaml")
os.makedirs("results/figures",exist_ok=True)
K={"gender":2,"age":8,"race":6}; R={"gender":2,"age":3,"race":5}
def fit(E,l,k):
    g=sorted(l.unique().tolist()); c=torch.stack([E[l==x].mean(0) for x in g])
    _,_,Vt=torch.linalg.svd(c-c.mean(0,keepdim=True),full_matrices=False); return Vt.T[:,:k]
def proj(E,B):
    d=E-(E@B)@B.T; return d/d.norm(dim=-1,keepdim=True).clamp(min=1e-8)
for attr in ["gender","age","race"]:
    m=FairCLIP(model_name="ViT-B/32",device="cuda")
    m.load(f"results/checkpoints/fairface/ViT-B_32/{attr}/seed42/best_model.pt"); m.eval()
    ds=FaceDataset(cfg.fairface.manifest,split="val",train=False)
    ld=DataLoader(ds,batch_size=64,shuffle=False,collate_fn=collate_dict,num_workers=2)
    img,labs=[],[]
    with torch.no_grad():
        for b in ld:
            img.append(m.backbone.encode_images(b["image"].to("cuda")).cpu())
            labs.append(b[attr].clone().detach())
    img=torch.cat(img).float(); labs=torch.cat(labs); img=img/img.norm(dim=-1,keepdim=True)
    ei=img.clone()
    for _ in range(R[attr]):
        B=fit(ei,labs,K[attr]); ei=proj(ei,B)
    gs=sorted(labs.unique().tolist())
    def spread(E):
        c=torch.stack([E[labs==g].mean(0) for g in gs]); mu=c.mean(0)
        return float(np.mean([ (x-mu).norm().item() for x in c]))
    print(f"{attr}: centroid spread before={spread(img):.4f} after={spread(ei):.4f} "
          f"reduction={100*(spread(img)-spread(ei))/spread(img):.1f}%")
    d=[(img[labs==g].mean(0)-ei[labs==g].mean(0)).norm().item() for g in gs]
    plt.figure(figsize=(6,4)); plt.bar(range(len(gs)),d)
    plt.xlabel("group"); plt.ylabel("centroid shift (L2)"); plt.title(f"{attr}: group centroid movement")
    plt.tight_layout(); plt.savefig(f"results/figures/centroid_shift_{attr}.png",dpi=200); plt.close()
    del m; torch.cuda.empty_cache()
PY
echo done > gaps_done.txt
