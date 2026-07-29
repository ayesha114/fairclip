#!/usr/bin/env bash
cd /home/ayeshakalsoom/Downloads/thesis
source /home/ayeshakalsoom/Downloads/fairclip-thesis/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1) frozen+proj Flickr for B/16, L/14
python3 - << 'PY' > flickr_frozen_all.txt 2>&1
import torch
from models.fairclip import FairCLIP
from evaluation.retrieval import evaluate_retrieval
for bb,bbt in [("ViT-B/16","ViT-B_16"),("ViT-L/14","ViT-L_14")]:
    for attr in ["gender","age","race"]:
        t = FairCLIP(model_name=bb, device="cuda")
        t.load(f"results/checkpoints/fairface/{bbt}/{attr}/seed42/best_model.pt")
        f = FairCLIP(model_name=bb, device="cuda")
        f.bias_remover.set_bias_subspace(t.bias_discoverer.image_bias_directions,
                                         t.bias_discoverer.text_bias_directions)
        f._bias_subspace_fitted = True; f.eval()
        with torch.no_grad():
            r = evaluate_retrieval(f, dataset="flickr30k", max_samples=1000, device="cuda")
        print(f"frozen+proj {bbt} {attr}: TR@5={r.get('TR@5',0):.1f} IR@5={r.get('IR@5',0):.1f}")
        del t,f; torch.cuda.empty_cache()
PY

# 2) iterative-eval seeds 123,456 (B/32, final protocol mean/std)
python3 - << 'PY' > iter_seeds.txt 2>&1
import torch
from omegaconf import OmegaConf
from models.fairclip import FairCLIP
from data.datasets import FaceDataset, collate_dict
from torch.utils.data import DataLoader
from evaluation.metrics.all_metrics import maxskew_ndkl_zhang, ZHANG_NEUTRAL_QUERIES
cfg = OmegaConf.load("configs/datasets/local_paths.yaml")
K = {"gender":2,"race":6,"age":8}
for attr in ["gender","age","race"]:
    for seed in [123,456]:
        m = FairCLIP(model_name="ViT-B/32", device="cuda")
        m.load(f"results/checkpoints/fairface/ViT-B_32/{attr}/seed{seed}/best_model.pt"); m.eval()
        ds = FaceDataset(cfg.fairface.manifest, split="val", train=False)
        ld = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_dict, num_workers=2)
        img,labs=[],[]
        with torch.no_grad():
            for b in ld:
                img.append(m.backbone.encode_images(b["image"].to("cuda")).cpu())
                labs.append(b[attr].clone().detach())
        img=torch.cat(img).float(); labs=torch.cat(labs)
        img=img/img.norm(dim=-1,keepdim=True)
        txt=m.encode_text(ZHANG_NEUTRAL_QUERIES).cpu().float(); txt=txt/txt.norm(dim=-1,keepdim=True)
        def fit(E,l,k):
            g=sorted(l.unique().tolist()); c=torch.stack([E[l==x].mean(0) for x in g])
            _,_,Vt=torch.linalg.svd(c-c.mean(0,keepdim=True),full_matrices=False); return Vt.T[:,:k]
        def proj(E,B):
            d=E-(E@B)@B.T; return d/d.norm(dim=-1,keepdim=True).clamp(min=1e-8)
        ei,et=img.clone(),txt.clone(); best=(9e9,9e9,0)
        for r in range(1,7):
            B=fit(ei,labs,K[attr]); ei=proj(ei,B); et=proj(et,B)
            ms,nd=maxskew_ndkl_zhang(et,ei,labs,k_skew=1000,k_ndkl=1000)
            if ms<best[0]: best=(ms,nd,r)
        print(f"{attr} seed{seed}: MaxSkew={best[0]:.3f} NDKL={best[1]:.3f} (round {best[2]})")
        del m; torch.cuda.empty_cache()
PY
echo done > day_queue_done.txt
