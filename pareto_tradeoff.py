"""Empirical Pareto front: as projection depth increases, retrieval fairness
(MaxSkew, down) improves but classification accuracy (down) degrades.
This is the evidence behind the trade-off theorem."""
import torch, csv
from omegaconf import OmegaConf
from models.fairclip import FairCLIP
from data.datasets import FaceDataset, collate_dict
from torch.utils.data import DataLoader
from evaluation.metrics.all_metrics import maxskew_ndkl_zhang, ZHANG_NEUTRAL_QUERIES
cfg=OmegaConf.load("configs/datasets/local_paths.yaml")
ATTR="race"
RACE=["a photo of a White person","a photo of a Black person","a photo of a Latino Hispanic person",
 "a photo of an East Asian person","a photo of a Southeast Asian person","a photo of an Indian person",
 "a photo of a Middle Eastern person"]
m=FairCLIP(model_name="ViT-B/32",device="cuda")
m.load(f"results/checkpoints/fairface/ViT-B_32/{ATTR}/seed42/best_model.pt"); m.eval()
ds=FaceDataset(cfg.fairface.manifest,split="val",train=False)
ld=DataLoader(ds,batch_size=64,shuffle=False,collate_fn=collate_dict,num_workers=2)
img,labs=[],[]
with torch.no_grad():
    for b in ld:
        img.append(m.backbone.encode_images(b["image"].to("cuda")).cpu())
        labs.append(b[ATTR].clone().detach())
img=torch.cat(img).float(); labs=torch.cat(labs); img=img/img.norm(dim=-1,keepdim=True)
cls=m.encode_text(RACE).cpu().float(); cls=cls/cls.norm(dim=-1,keepdim=True)
neu=m.encode_text(ZHANG_NEUTRAL_QUERIES).cpu().float(); neu=neu/neu.norm(dim=-1,keepdim=True)
def fit(E,l,k=6):
    g=sorted(l.unique().tolist()); c=torch.stack([E[l==x].mean(0) for x in g])
    _,_,Vt=torch.linalg.svd(c-c.mean(0,keepdim=True),full_matrices=False); return Vt.T[:,:k]
def proj(E,B):
    d=E-(E@B)@B.T; return d/d.norm(dim=-1,keepdim=True).clamp(min=1e-8)
print("round | Acc(class) | MaxSkew(retr) | NDKL")
ei,ecls,eneu=img.clone(),cls.clone(),neu.clone()
rows=[]
# round 0 = no projection
acc=((ei@ecls.T).argmax(1)==labs).float().mean().item()
ms,nd=maxskew_ndkl_zhang(eneu,ei,labs,k_skew=1000,k_ndkl=1000)
print(f"  0   |  {acc:.3f}    |    {ms:.3f}     | {nd:.3f}"); rows.append([0,acc,ms,nd])
for r in range(1,11):
    B=fit(ei,labs); ei=proj(ei,B); ecls=proj(ecls,B); eneu=proj(eneu,B)
    acc=((ei@ecls.T).argmax(1)==labs).float().mean().item()
    ms,nd=maxskew_ndkl_zhang(eneu,ei,labs,k_skew=1000,k_ndkl=1000)
    print(f"  {r}   |  {acc:.3f}    |    {ms:.3f}     | {nd:.3f}"); rows.append([r,acc,ms,nd])
with open("results/tables/pareto_tradeoff.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["round","class_acc","maxskew","ndkl"]); w.writerows(rows)
print("saved results/tables/pareto_tradeoff.csv")
