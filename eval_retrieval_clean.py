import argparse, os, torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from models.fairclip import FairCLIP
from models.bias_subspace import BiasSubspaceDiscoverer
from data.datasets import FaceDataset, collate_dict
from evaluation.metrics.all_metrics import maxskew_ndkl_zhang, ZHANG_NEUTRAL_QUERIES

ZHANG = {"gender":{"MS":0.090,"NDKL":0.030},"age":{"MS":0.572,"NDKL":0.364},"race":{"MS":0.353,"NDKL":0.125}}

def encode(m, cfg, ds_name, attr, split, dev, bb, cap=None):
    ds = FaceDataset(cfg[ds_name].manifest, split=split, train=False)
    ebs = 16 if bb in ("ViT-L/14","ViT-H/14") else 64
    ld = DataLoader(ds, batch_size=ebs, shuffle=False, collate_fn=collate_dict, num_workers=2)
    E,L=[],[]
    with torch.no_grad():
        for b in ld:
            E.append(m.backbone.encode_images(b["image"].to(dev)).cpu()); L.append(b[attr].clone().detach())
            if cap and sum(x.shape[0] for x in E)>=cap: break
    E=torch.cat(E).float(); L=torch.cat(L)
    if cap: E,L=E[:cap],L[:cap]
    return E/E.norm(dim=-1,keepdim=True), L

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--attribute",required=True); ap.add_argument("--backbone",default="ViT-B/32")
    ap.add_argument("--dataset",default="fairface"); ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--rounds",type=int,default=8); ap.add_argument("--n_dirs",type=int,default=5); ap.add_argument("--model_path",default=None)
    ap.add_argument("--train_cap",type=int,default=20000)
    a=ap.parse_args()
    cfg=OmegaConf.load("configs/datasets/local_paths.yaml")
    dev="cuda" if torch.cuda.is_available() else "cpu"
    bb=a.backbone.replace("/","_")
    if a.model_path:
        ckpt=a.model_path
    else:
     for p in [f"results/checkpoints/{a.dataset}/{bb}/{a.attribute}/seed{a.seed}/best_model.pt",
              f"results/checkpoints/{bb}/{a.attribute}/seed{a.seed}/best_model.pt",
              f"results/checkpoints_fix1/{a.dataset}/{bb}/{a.attribute}/seed{a.seed}/best_model.pt"]:
        if os.path.exists(p): ckpt=p; break
     else: raise FileNotFoundError("no checkpoint")
    m=FairCLIP(model_name=a.backbone,device=dev); m.load(ckpt); m.eval()
    neu=m.backbone.encode_text(ZHANG_NEUTRAL_QUERIES).cpu().float(); neu=neu/neu.norm(dim=-1,keepdim=True)
    print(f"[{a.attribute} {a.backbone}] encoding train+val...",flush=True)
    tr,trl=encode(m,cfg,a.dataset,a.attribute,"train",dev,a.backbone,cap=a.train_cap)
    va,val=encode(m,cfg,a.dataset,a.attribute,"val",dev,a.backbone)
    ms0,nd0=maxskew_ndkl_zhang(neu,va,val,k_skew=1000,k_ndkl=1000)
    tr2,va2,neu2=tr.clone(),va.clone(),neu.clone()
    def fit(E,l,k):
        g=sorted(l.unique().tolist()); c=torch.stack([E[l==x].mean(0) for x in g])
        _,_,Vt=torch.linalg.svd(c-c.mean(0,keepdim=True),full_matrices=False); return Vt.T[:,:k]
    def proj(E,B): d=E-(E@B)@B.T; return d/d.norm(dim=-1,keepdim=True).clamp(min=1e-8)
    for _ in range(a.rounds):
        B=fit(tr2,trl,a.n_dirs); tr2=proj(tr2,B); va2=proj(va2,B); neu2=proj(neu2,B)
    ms1,nd1=maxskew_ndkl_zhang(neu2,va2,val,k_skew=1000,k_ndkl=1000)
    z=ZHANG.get(a.attribute,{"MS":9,"NDKL":9})
    beats="YES" if (ms1<z["MS"] and nd1<z["NDKL"]) else ("PARTIAL" if (ms1<z["MS"] or nd1<z["NDKL"]) else "NO")
    print(f"\n=== CLEAN (fit-on-train, no INLP): {a.attribute} {a.backbone} ===")
    print(f"Before proj : MaxSkew={ms0:.3f} NDKL={nd0:.3f}")
    print(f"FairCLIP    : MaxSkew={ms1:.3f} NDKL={nd1:.3f}")
    print(f"Zhang       : MaxSkew={z['MS']:.3f} NDKL={z['NDKL']:.3f}")
    print(f"Beats Zhang : {beats}")
    import csv
    os.makedirs("results/tables",exist_ok=True)
    with open(f"results/tables/retrieval_clean_{a.attribute}_{bb}_seed{a.seed}.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["attribute","backbone","seed","MaxSkew","NDKL","Zhang_MS","Zhang_NDKL","beats"])
        w.writerow([a.attribute,a.backbone,a.seed,f"{ms1:.4f}",f"{nd1:.4f}",z["MS"],z["NDKL"],beats])
    print("saved csv")

if __name__=="__main__": main()
