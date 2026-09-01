"""Complete evaluation: ALL metrics from a trained checkpoint. No retraining."""
import argparse, torch, numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from models.fairclip import FairCLIP
from data.datasets import FaceDataset, collate_dict
from evaluation.metrics.all_metrics import (
    compute_dpg_occupation, compute_eod_occupation, compute_rbs,
    compute_accuracy, compute_able_correct, maxskew_ndkl_zhang,
    OCCUPATION_PROMPTS, ZHANG_NEUTRAL_QUERIES)

ATTR_PROMPTS = {
    "gender": ["a photo of a male person","a photo of a female person"],
    "age": ["a photo of a person aged 0 to 2","a photo of a person aged 3 to 9",
            "a photo of a person aged 10 to 19","a photo of a person aged 20 to 29",
            "a photo of a person aged 30 to 39","a photo of a person aged 40 to 49",
            "a photo of a person aged 50 to 59","a photo of a person aged 60 to 69",
            "a photo of a person more than 70 years old"],
    "race": ["a photo of a White person","a photo of a Black person",
             "a photo of a Latino Hispanic person","a photo of an East Asian person",
             "a photo of a Southeast Asian person","a photo of an Indian person",
             "a photo of a Middle Eastern person"],
}

def encode_split(m, manifest, attr, split, dev, bb, cap=None):
    ds = FaceDataset(manifest, split=split, train=False)
    ebs = 16 if bb in ("ViT-L/14","ViT-H/14") else 64
    ld = DataLoader(ds, batch_size=ebs, shuffle=False, collate_fn=collate_dict, num_workers=2)
    E,L=[],[]
    with torch.no_grad():
        for b in ld:
            E.append(m.backbone.encode_images(b["image"].to(dev)).cpu())
            L.append(b[attr].clone().detach())
            if cap and sum(x.shape[0] for x in E)>=cap: break
    E=torch.cat(E).float(); L=torch.cat(L)
    if cap: E,L=E[:cap],L[:cap]
    E=E/E.norm(dim=-1,keepdim=True)
    return E,L

def fit_subspace(embs, labels, k):
    groups=sorted([g.item() for g in labels.unique() if g.item()!=-1])
    cent=torch.stack([embs[labels==g].mean(0) for g in groups])
    cent=cent-cent.mean(0,keepdim=True)
    U,S,Vt=torch.linalg.svd(cent,full_matrices=False)
    k_eff=min(k,len(groups)-1)
    return Vt.T[:,:k_eff]

def project_out(embs,B):
    d=embs-(embs@B)@B.T
    return d/d.norm(dim=-1,keepdim=True).clamp(min=1e-8)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--attribute",required=True)
    ap.add_argument("--backbone",default="ViT-B/32")
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--n_dirs",type=int,default=8)
    ap.add_argument("--baseline",action="store_true")
    ap.add_argument("--train_cap",type=int,default=20000)
    ap.add_argument("--model_path",default=None)
    ap.add_argument("--dataset",default="fairface")
    a=ap.parse_args()
    cfg=OmegaConf.load("configs/datasets/local_paths.yaml")
    manifest = cfg[a.dataset].manifest if a.dataset in cfg else cfg.fairface.manifest
    dev="cuda" if torch.cuda.is_available() else "cpu"
    m=FairCLIP(model_name=a.backbone,device=dev)
    if not a.baseline:
        ck=a.model_path
        m.load(ck)
    m.eval()
    va,val=encode_split(m,manifest,a.attribute,"val",dev,a.backbone)
    occ=m.backbone.encode_text(OCCUPATION_PROMPTS).cpu().float(); occ=occ/occ.norm(dim=-1,keepdim=True)
    at=m.backbone.encode_text(ATTR_PROMPTS[a.attribute]).cpu().float(); at=at/at.norm(dim=-1,keepdim=True)
    neu=m.backbone.encode_text(list(ZHANG_NEUTRAL_QUERIES)).cpu().float(); neu=neu/neu.norm(dim=-1,keepdim=True)
    if a.baseline:
        img=va; occ_u=occ; at_u=at; tag="BASELINE"
    else:
        tr,trl=encode_split(m,manifest,a.attribute,"train",dev,a.backbone,cap=a.train_cap)
        B=fit_subspace(tr,trl,a.n_dirs)
        img=project_out(va,B); occ_u=project_out(occ,B); at_u=project_out(at,B); neu=project_out(neu,B)
        tag="FairCLIP"
    dpg=compute_dpg_occupation(img,occ_u,val)
    eod=compute_eod_occupation(img,occ_u,val)
    rbs=compute_rbs(img,val)
    valid=val!=-1
    pred=(img[valid].float()@at_u.float().T).argmax(dim=1)
    gt=val[valid]
    acc=compute_accuracy(pred,gt)
    f1s=[]
    for c in gt.unique():
        p=(pred==c); g=(gt==c)
        tp=(p&g).sum().item(); fp=(p&~g).sum().item(); fn=(~p&g).sum().item()
        prec=tp/(tp+fp) if tp+fp>0 else 0; rec=tp/(tp+fn) if tp+fn>0 else 0
        f1s.append(2*prec*rec/(prec+rec) if prec+rec>0 else 0)
    f1=sum(f1s)/len(f1s) if f1s else 0
    ms,nd=maxskew_ndkl_zhang(neu,img,val,k_skew=1000,k_ndkl=1000)
    able=compute_able_correct(img,at_u,val,k=50)
    able_v=able["able"] if isinstance(able,dict) else able
    print(f"{tag} {a.dataset} {a.backbone} {a.attribute} s{a.seed}: Acc={acc:.4f} F1={f1:.4f} DPG={dpg:.4f} EOD={eod:.4f} RBS={rbs:.2e} MaxSkew={ms:.3f} NDKL={nd:.3f} ABLE={able_v:.2f}")

if __name__=="__main__":
    main()
