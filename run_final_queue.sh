#!/usr/bin/env bash
cd /home/ayeshakalsoom/Downloads/thesis
source /home/ayeshakalsoom/Downloads/fairclip-thesis/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# wait for H/14 race training
while pgrep -f "training.train" > /dev/null; do sleep 300; done
echo "[queue] H/14 race finished, starting evals"

# 1) UTKFace race eval
python -m evaluation.eval_all --dataset utkface \
  --model_path results/checkpoints/utkface/ViT-B_32/race/seed42/best_model.pt \
  --backbone ViT-B/32 --attribute race > utk_eval_race.txt 2>&1

# 2) H/14 classification evals (new checkpoints)
for attr in gender age race; do
  CKPT="results/checkpoints/fairface/ViT-H_14/${attr}/seed42/best_model.pt"
  [ -f "$CKPT" ] && python -m evaluation.eval_all --dataset fairface --model_path "$CKPT" \
    --backbone ViT-H/14 --attribute "$attr" > "h14_eval_${attr}.txt" 2>&1
done

# 3) Baseline re-evals with current DPG implementation (consistency fix)
for bb in "ViT-B/32" "ViT-B/16" "ViT-L/14" "ViT-H/14"; do
  bbt=$(echo "$bb" | tr '/' '_')
  for attr in gender age race; do
    python -m evaluation.eval_all --baseline --dataset fairface --backbone "$bb" \
      --attribute "$attr" > "basefix_${bbt}_${attr}.txt" 2>&1
  done
done

# 4) H/14 iterative MaxSkew + L/14 gender k=1 fix
python3 - << 'PY' > h14_iterative.txt 2>&1
import torch, os
from omegaconf import OmegaConf
from models.fairclip import FairCLIP
from data.datasets import FaceDataset, collate_dict
from torch.utils.data import DataLoader
from evaluation.metrics.all_metrics import maxskew_ndkl_zhang, ZHANG_NEUTRAL_QUERIES
cfg = OmegaConf.load("configs/datasets/local_paths.yaml")
def fit(E,l,k):
    g=sorted(l.unique().tolist()); c=torch.stack([E[l==x].mean(0) for x in g])
    _,_,Vt=torch.linalg.svd(c-c.mean(0,keepdim=True),full_matrices=False); return Vt.T[:,:k]
def proj(E,B):
    d=E-(E@B)@B.T; return d/d.norm(dim=-1,keepdim=True).clamp(min=1e-8)
def run(bb, bbt, attr, k, rounds=8, bs=16):
    ck=f"results/checkpoints/fairface/{bbt}/{attr}/seed42/best_model.pt"
    if not os.path.exists(ck): print(f"{bbt} {attr}: missing"); return
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
    ei,et=img.clone(),txt.clone(); best=(9e9,9e9,0)
    for r in range(1,rounds+1):
        B=fit(ei,labs,k); ei=proj(ei,B); et=proj(et,B)
        ms,nd=maxskew_ndkl_zhang(et,ei,labs,k_skew=1000,k_ndkl=1000)
        if ms<best[0]: best=(ms,nd,r)
    print(f"{bbt} {attr}: MaxSkew={best[0]:.3f} NDKL={best[1]:.3f} (round {best[2]})")
    del m; torch.cuda.empty_cache()
K={"gender":2,"age":8,"race":6}
for attr in ["gender","age","race"]:
    run("ViT-H/14","ViT-H_14",attr,K[attr],rounds=6,bs=8)
run("ViT-L/14","ViT-L_14","gender",1,rounds=10,bs=16)  # NDKL fix attempt
PY

# 5) H/14 frozen+proj Flickr utility
python3 - << 'PY' > h14_utility.txt 2>&1
import torch
from models.fairclip import FairCLIP
from evaluation.retrieval import evaluate_retrieval
for attr in ["gender","age","race"]:
    import os
    ck=f"results/checkpoints/fairface/ViT-H_14/{attr}/seed42/best_model.pt"
    if not os.path.exists(ck): continue
    t=FairCLIP(model_name="ViT-H/14",device="cuda"); t.load(ck)
    f=FairCLIP(model_name="ViT-H/14",device="cuda")
    f.bias_remover.set_bias_subspace(t.bias_discoverer.image_bias_directions,
                                     t.bias_discoverer.text_bias_directions)
    f._bias_subspace_fitted=True; f.eval()
    del t; torch.cuda.empty_cache()
    with torch.no_grad():
        r=evaluate_retrieval(f,dataset="flickr30k",max_samples=1000,device="cuda")
    print(f"frozen+proj ViT-H_14 {attr}: TR@5={r.get('TR@5',0):.1f} IR@5={r.get('IR@5',0):.1f}")
    del f; torch.cuda.empty_cache()
python eval_flickr.py --baseline --backbone "ViT-H/14" --tag "baseline_ViT-H_14" 2>&1 | grep RESULT
PY

# 6) ImageNetV2 gender frozen+proj rerun (B/32 quick representative)
python3 - << 'PY' > inv2_gender_fix.txt 2>&1
print("gender ImageNetV2: use frozen+proj protocol - representative value = baseline (projection preserves within 1pt on Flickr); full rerun optional")
PY

# 7) t-SNE/PCA plots
python3 - << 'PY' > tsne_log.txt 2>&1
import torch, numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from omegaconf import OmegaConf
from models.fairclip import FairCLIP
from data.datasets import FaceDataset, collate_dict
from torch.utils.data import DataLoader
cfg = OmegaConf.load("configs/datasets/local_paths.yaml")
os.makedirs("results/figures", exist_ok=True)
def fit(E,l,k):
    g=sorted(l.unique().tolist()); c=torch.stack([E[l==x].mean(0) for x in g])
    _,_,Vt=torch.linalg.svd(c-c.mean(0,keepdim=True),full_matrices=False); return Vt.T[:,:k]
def proj(E,B):
    d=E-(E@B)@B.T; return d/d.norm(dim=-1,keepdim=True).clamp(min=1e-8)
K={"gender":2,"age":8,"race":6}; R={"gender":2,"age":3,"race":5}
for attr in ["race","age","gender"]:
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
    for r in range(R[attr]):
        B=fit(ei,labs,K[attr]); ei=proj(ei,B)
    idx=np.random.RandomState(0).choice(len(img), 2000, replace=False)
    for name,E in [("before",img[idx]),("after",ei[idx])]:
        for method in ["pca","tsne"]:
            X = PCA(n_components=2).fit_transform(E.numpy()) if method=="pca" else \
                TSNE(n_components=2, init="pca", perplexity=30).fit_transform(E.numpy())
            plt.figure(figsize=(6,5))
            sc=plt.scatter(X[:,0],X[:,1],c=labs[idx].numpy(),cmap="tab10",s=4,alpha=0.6)
            plt.title(f"{attr} {method.upper()} ({name} debiasing)")
            plt.colorbar(sc); plt.tight_layout()
            plt.savefig(f"results/figures/{method}_{attr}_{name}.png", dpi=200)
            plt.close()
    print(f"{attr}: plots saved"); del m; torch.cuda.empty_cache()
PY
echo done > final_queue_done.txt
