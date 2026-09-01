"""t-SNE of image embeddings before/after debiasing (proposal RO4, cf. Zhang Fig.5)."""
import torch, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from models.fairclip import FairCLIP
from data.datasets import FaceDataset, collate_dict

cfg=OmegaConf.load("configs/datasets/local_paths.yaml")
N=2000; BB="ViT-B/32"
NAMES={"gender":["Male","Female"],
 "age":["0-2","3-9","10-19","20-29","30-39","40-49","50-59","60-69","70+"],
 "race":["White","Black","Latino","East Asian","SE Asian","Indian","Mid. Eastern"]}

def enc(m,attr,split,cap):
    ds=FaceDataset(cfg.fairface.manifest,split=split,train=False)
    ld=DataLoader(ds,batch_size=64,shuffle=False,collate_fn=collate_dict,num_workers=2)
    E,L=[],[]
    with torch.no_grad():
        for b in ld:
            E.append(m.backbone.encode_images(b["image"].to("cuda")).cpu())
            L.append(b[attr].clone().detach())
            if sum(x.shape[0] for x in E)>=cap: break
    E=torch.cat(E)[:cap].float(); L=torch.cat(L)[:cap]
    return E/E.norm(dim=-1,keepdim=True), L

fig,axes=plt.subplots(2,3,figsize=(16,10))
for j,attr in enumerate(["gender","age","race"]):
    ck=f"results/checkpoints_procfix/fairface/ViT-B_32/{attr}/seed42/best_model.pt"
    m=FairCLIP(model_name=BB,device="cuda"); m.load(ck); m.eval()
    va,val=enc(m,attr,"val",N)
    tr,trl=enc(m,attr,"train",20000)
    g=sorted([x for x in trl.unique().tolist() if x!=-1])
    c=torch.stack([tr[trl==x].mean(0) for x in g]); c=c-c.mean(0,keepdim=True)
    _,_,Vt=torch.linalg.svd(c,full_matrices=False); B=Vt.T[:,:min(8,len(g)-1)]
    d=va-(va@B)@B.T; after=d/d.norm(dim=-1,keepdim=True).clamp(min=1e-8)
    keep=val>=0
    X0=va[keep].numpy(); X1=after[keep].numpy(); y=val[keep].numpy()
    for i,(X,ttl) in enumerate([(X0,"Before (baseline CLIP)"),(X1,"After (FairCLIP)")]):
        Z=TSNE(n_components=2,perplexity=30,init="pca",random_state=42).fit_transform(X)
        ax=axes[i][j]
        for gi in sorted(set(y.tolist())):
            s=y==gi
            lbl=NAMES[attr][gi] if gi<len(NAMES[attr]) else str(gi)
            ax.scatter(Z[s,0],Z[s,1],s=4,alpha=.6,label=lbl)
        ax.set_title(f"{attr} — {ttl}",fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        if i==0: ax.legend(fontsize=6,markerscale=2,loc="best")
    del m,tr,trl; torch.cuda.empty_cache()
    print(f"done {attr}",flush=True)
plt.tight_layout()
plt.savefig("FINAL_RESULTS/tsne_before_after.png",dpi=180,bbox_inches="tight")
print("SAVED FINAL_RESULTS/tsne_before_after.png",flush=True)
