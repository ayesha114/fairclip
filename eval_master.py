import os, csv, argparse, numpy as np, torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from models.fairclip import FairCLIP
from data.datasets import FaceDataset, collate_dict
from evaluation.retrieval import evaluate_retrieval
from evaluation.metrics.all_metrics import (
    compute_dpg_occupation, compute_eod_occupation, compute_rbs,
    maxskew_ndkl_zhang, OCCUPATION_PROMPTS, ZHANG_NEUTRAL_QUERIES)

OUT = "results/MASTER_RESULTS.csv"
HDR = ["group","config","backbone","attribute","seed","Acc","F1","DPG","EOD","RBS","MaxSkew","NDKL","ABLE","TR5","IR5"]
AP = {
 "gender":["a photo of a male person","a photo of a female person"],
 "age":["a photo of a person aged 0-2","a photo of a person aged 3-9","a photo of a person aged 10-19",
        "a photo of a person aged 20-29","a photo of a person aged 30-39","a photo of a person aged 40-49",
        "a photo of a person aged 50-59","a photo of a person aged 60-69","a photo of a person aged 70 or older"],
 "race":["a photo of a White person","a photo of a Black person","a photo of a Latino Hispanic person",
         "a photo of an East Asian person","a photo of a Southeast Asian person","a photo of an Indian person",
         "a photo of a Middle Eastern person"]}

def done_keys():
    if not os.path.exists(OUT): return set()
    return {tuple(r[:5]) for r in list(csv.reader(open(OUT)))[1:]}

def append(row):
    new = not os.path.exists(OUT)
    with open(OUT,"a",newline="") as f:
        w=csv.writer(f)
        if new: w.writerow(HDR)
        w.writerow(row)

def encode(m,man,attr,split,bb,cap=None,rc="race_idx"):
    ds=FaceDataset(man,split=split,train=False,race_column=rc)
    bs=8 if bb=="ViT-H/14" else (16 if bb=="ViT-L/14" else 64)
    ld=DataLoader(ds,batch_size=bs,shuffle=False,collate_fn=collate_dict,num_workers=2)
    E,L=[],[]
    with torch.no_grad():
        for b in ld:
            E.append(m.backbone.encode_images(b["image"].to("cuda")).cpu())
            L.append(b[attr].clone().detach())
            if cap and sum(x.shape[0] for x in E)>=cap: break
    E=torch.cat(E).float(); L=torch.cat(L)
    if cap: E,L=E[:cap],L[:cap]
    return E/E.norm(dim=-1,keepdim=True), L

def fit_sub(E,L,k):
    g=sorted([x.item() for x in L.unique() if x.item()!=-1])
    c=torch.stack([E[L==x].mean(0) for x in g]); c=c-c.mean(0,keepdim=True)
    _,_,Vt=torch.linalg.svd(c,full_matrices=False)
    return Vt.T[:,:min(k,len(g)-1)]

def proj(E,B):
    d=E-(E@B)@B.T
    return d/d.norm(dim=-1,keepdim=True).clamp(min=1e-8)

def mf1(pred,gt):
    f=[]
    for c in gt.unique():
        p=(pred==c); g=(gt==c)
        tp=(p&g).sum().item(); fp=(p&(~g)).sum().item(); fn=((~p)&g).sum().item()
        pr=tp/(tp+fp) if tp+fp else 0.0
        rc=tp/(tp+fn) if tp+fn else 0.0
        f.append(2*pr*rc/(pr+rc) if pr+rc else 0.0)
    return float(np.mean(f)) if f else 0.0

def run_cell(grp,cfgn,bb,attr,seed,ck,man,base,nd,rc,do_r):
    m=FairCLIP(model_name=bb,device="cuda")
    if not base: m.load(ck)
    m.eval()
    va,val=encode(m,man,attr,"val",bb,rc=rc)
    occ=m.backbone.encode_text(OCCUPATION_PROMPTS).cpu().float(); occ=occ/occ.norm(dim=-1,keepdim=True)
    at=m.backbone.encode_text(AP[attr]).cpu().float(); at=at/at.norm(dim=-1,keepdim=True)
    neu=m.backbone.encode_text(list(ZHANG_NEUTRAL_QUERIES)).cpu().float(); neu=neu/neu.norm(dim=-1,keepdim=True)
    imgd=txtd=None
    if base:
        img,occu,atu,neuu=va,occ,at,neu
    else:
        tr,trl=encode(m,man,attr,"train",bb,cap=20000,rc=rc)
        B=fit_sub(tr,trl,nd); del tr,trl
        img=proj(va,B); occu=proj(occ,B); atu=proj(at,B); neuu=proj(neu,B)
        d1=m.bias_discoverer.image_bias_directions
        d2=m.bias_discoverer.text_bias_directions
        if d1 is not None: imgd=d1.detach().cpu().clone()
        if d2 is not None: txtd=d2.detach().cpu().clone()
    dpg=compute_dpg_occupation(img,occu,val)
    eod=compute_eod_occupation(img,occu,val)
    rbs=compute_rbs(img,val)
    v=val!=-1
    pred=(img[v].float()@atu.float().T).argmax(1); gt=val[v]
    acc=float((pred==gt).float().mean().item()); f1=mf1(pred,gt)
    ms,nk=maxskew_ndkl_zhang(neuu,img,val,k_skew=1000,k_ndkl=1000)
    able=2.0/(1.0/max(acc,1e-6)+1.0/max(float(np.exp(-ms)),1e-6))*100.0
    del m,va,img,occ,at,neu; torch.cuda.empty_cache()
    tr5=ir5=""
    if do_r:
        fm=FairCLIP(model_name=bb,device="cuda")
        if imgd is not None:
            fm.bias_remover.set_bias_subspace(imgd.cuda(),txtd.cuda()); fm._bias_subspace_fitted=True
        fm.eval()
        with torch.no_grad():
            r=evaluate_retrieval(fm,dataset="flickr30k",max_samples=1000,device="cuda")
        tr5=round(r.get("TR@5",0),1); ir5=round(r.get("IR@5",0),1)
        del fm; torch.cuda.empty_cache()
    return [grp,cfgn,bb,attr,str(seed),round(acc,4),round(f1,4),round(dpg,4),round(eod,4),
            "%.2e"%rbs,round(ms,3),round(nk,3),round(able,2),tr5,ir5]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",default="fairface")
    ap.add_argument("--n_dirs",type=int,default=8)
    ap.add_argument("--no_retrieval",action="store_true")
    a=ap.parse_args()
    cfg=OmegaConf.load("configs/datasets/local_paths.yaml")
    man=cfg[a.dataset].manifest
    rc="race_idx_7" if a.dataset=="utkface" else "race_idx"
    os.makedirs("results",exist_ok=True)
    dk=done_keys()
    BB=["ViT-B/32","ViT-B/16","ViT-L/14","ViT-H/14"]; AT=["gender","age","race"]
    jobs=[]
    for bb in BB:
        for t in AT: jobs.append(("BASELINE","baseline",bb,t,42,None,True))
    for s in [42,123,456]:
        for bb in BB:
            for t in AT:
                jobs.append(("MULTISEED","fairclip",bb,t,s,
                  "results/checkpoints_procfix/fairface/%s/%s/seed%d/best_model.pt"%(bb.replace("/","_"),t,s),False))
    for n in ["no_fairloss","no_temp","no_trainproj","no_procrustes","no_subspace"]:
        for t in AT:
            jobs.append(("ABLATION",n,"ViT-B/32",t,42,
              "results/checkpoints_ablfix_%s/fairface/ViT-B_32/%s/seed42/best_model.pt"%(n,t),False))
    for grp,cfgn,bb,t,s,ck,base in jobs:
        key=(grp,cfgn,bb,t,str(s))
        if key in dk: print("SKIP",key,flush=True); continue
        if (not base) and not os.path.exists(ck): print("MISS",key,flush=True); continue
        try:
            row=run_cell(grp,cfgn,bb,t,s,ck,man,base,a.n_dirs,rc,not a.no_retrieval)
            append(row); print("DONE",key,row[5:],flush=True)
        except Exception as e:
            print("FAIL",key,repr(e)[:180],flush=True); torch.cuda.empty_cache()
    print("ALL FINISHED",flush=True)

main()
