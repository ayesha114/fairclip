"""Comprehensive FairCLIP visualizations (FairFace + UTKFace) with error bars."""
import pandas as pd, numpy as np, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAB="results/tables"; FIG="results/figures"; os.makedirs(FIG,exist_ok=True)
plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.3})

def parse(s):
    """'0.299±0.039' -> (0.299, 0.039)"""
    if isinstance(s,(int,float)): return float(s),0.0
    m=re.match(r"([\d.]+)±([\d.]+)",str(s))
    return (float(m.group(1)),float(m.group(2))) if m else (float(s),0.0)

def save(fig,n): fig.savefig(f"{FIG}/{n}",dpi=150,bbox_inches="tight"); plt.close(fig); print("saved",n)

# ---------- FairFace data ----------
clsf=pd.read_csv(f"{TAB}/COMPARE_classification.csv")
retr=pd.read_csv(f"{TAB}/COMPARE_retrieval.csv")
attrs=list(clsf["Attribute"]); x=np.arange(len(attrs)); w=0.35

# FIG A: FairFace DPG baseline vs FairCLIP
base=[parse(v)[0] for v in clsf["DPG_base"]]
ours=[parse(v)[0] for v in clsf["DPG_FairCLIP"]]
err =[parse(v)[1] for v in clsf["DPG_FairCLIP"]]
fig,ax=plt.subplots(figsize=(8,5))
ax.bar(x-w/2,base,w,label="Baseline CLIP",color="#9aa0a6")
ax.bar(x+w/2,ours,w,yerr=err,capsize=4,label="FairCLIP (ours)",color="#1f77b4")
ax.set_xticks(x);ax.set_xticklabels([a.capitalize() for a in attrs])
ax.set_ylabel("DPG (lower = fairer)")
ax.set_title("FairFace — Classification Fairness (DPG): Baseline vs FairCLIP")
ax.legend();save(fig,"A_fairface_dpg.png")

# FIG B: FairFace Accuracy baseline vs FairCLIP
base=[parse(v)[0] for v in clsf["Acc_base"]]
ours=[parse(v)[0] for v in clsf["Acc_FairCLIP"]]
err =[parse(v)[1] for v in clsf["Acc_FairCLIP"]]
fig,ax=plt.subplots(figsize=(8,5))
ax.bar(x-w/2,base,w,label="Baseline CLIP",color="#9aa0a6")
ax.bar(x+w/2,ours,w,yerr=err,capsize=4,label="FairCLIP (ours)",color="#2ca02c")
ax.set_xticks(x);ax.set_xticklabels([a.capitalize() for a in attrs])
ax.set_ylabel("Accuracy (higher = better)")
ax.set_title("FairFace — Utility Preserved (Accuracy): Baseline vs FairCLIP")
ax.legend();save(fig,"B_fairface_accuracy.png")

# FIG C: FairFace MaxSkew 3-way (Orig / FairCLIP / Zhang)
orig=[parse(v)[0] for v in retr["MS_OrigCLIP"]]
ours=[parse(v)[0] for v in retr["MS_FairCLIP"]]
err =[parse(v)[1] for v in retr["MS_FairCLIP"]]
zh  =[parse(v)[0] for v in retr["MS_Zhang"]]
w3=0.27
fig,ax=plt.subplots(figsize=(9,5))
ax.bar(x-w3,orig,w3,label="Original CLIP",color="#9aa0a6")
ax.bar(x,    zh, w3,label="Zhang et al.",color="#ff7f0e")
ax.bar(x+w3,ours,w3,yerr=err,capsize=4,label="FairCLIP (ours)",color="#1f77b4")
ax.set_xticks(x);ax.set_xticklabels([a.capitalize() for a in attrs])
ax.set_ylabel("MaxSkew (lower = fairer)")
ax.set_title("FairFace — Retrieval Fairness (MaxSkew): FairCLIP vs Zhang")
ax.legend();save(fig,"C_fairface_maxskew_3way.png")

# FIG D: FairFace NDKL 3-way
orig=[parse(v)[0] for v in retr["NDKL_OrigCLIP"]]
ours=[parse(v)[0] for v in retr["NDKL_FairCLIP"]]
err =[parse(v)[1] for v in retr["NDKL_FairCLIP"]]
zh  =[parse(v)[0] for v in retr["NDKL_Zhang"]]
fig,ax=plt.subplots(figsize=(9,5))
ax.bar(x-w3,orig,w3,label="Original CLIP",color="#9aa0a6")
ax.bar(x,    zh, w3,label="Zhang et al.",color="#ff7f0e")
ax.bar(x+w3,ours,w3,yerr=err,capsize=4,label="FairCLIP (ours)",color="#1f77b4")
ax.set_xticks(x);ax.set_xticklabels([a.capitalize() for a in attrs])
ax.set_ylabel("NDKL (lower = fairer)")
ax.set_title("FairFace — Retrieval Fairness (NDKL): FairCLIP vs Zhang")
ax.legend();save(fig,"D_fairface_ndkl_3way.png")

# ---------- UTKFace ----------
utk=pd.read_csv(f"{TAB}/UTKFACE_retrieval.csv")
ua=list(utk["Attribute"]); ux=np.arange(len(ua))
# FIG E: UTKFace MaxSkew & NDKL vs Zhang
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5))
for ax,ms_col,zh_col,err_col,lab in [
    (ax1,"MS_FairCLIP","MS_Zhang","MS","MaxSkew"),
    (ax2,"NDKL_FairCLIP","NDKL_Zhang","NDKL","NDKL")]:
    o=[parse(v)[0] for v in utk[ms_col]]; e=[parse(v)[1] for v in utk[ms_col]]
    z=[parse(v)[0] for v in utk[zh_col]]
    ax.bar(ux-0.2,z,0.4,label="Zhang et al.",color="#ff7f0e")
    ax.bar(ux+0.2,o,0.4,yerr=e,capsize=4,label="FairCLIP",color="#1f77b4")
    ax.set_xticks(ux);ax.set_xticklabels([a.capitalize() for a in ua])
    ax.set_ylabel(f"{lab} (lower = fairer)");ax.set_title(f"UTKFace — {lab}")
    ax.legend()
fig.suptitle("UTKFace (out-of-domain) — FairCLIP vs Zhang et al.",fontsize=13)
save(fig,"E_utkface_retrieval.png")

# FIG F: Combined dashboard
fig,axes=plt.subplots(2,2,figsize=(14,10))
# DPG
b=[parse(v)[0] for v in clsf["DPG_base"]];o=[parse(v)[0] for v in clsf["DPG_FairCLIP"]]
axes[0,0].bar(x-w/2,b,w,label="Baseline",color="#9aa0a6");axes[0,0].bar(x+w/2,o,w,label="FairCLIP",color="#1f77b4")
axes[0,0].set_title("FairFace DPG (lower=fairer)");axes[0,0].set_xticks(x);axes[0,0].set_xticklabels([a.capitalize() for a in attrs]);axes[0,0].legend()
# Acc
b=[parse(v)[0] for v in clsf["Acc_base"]];o=[parse(v)[0] for v in clsf["Acc_FairCLIP"]]
axes[0,1].bar(x-w/2,b,w,label="Baseline",color="#9aa0a6");axes[0,1].bar(x+w/2,o,w,label="FairCLIP",color="#2ca02c")
axes[0,1].set_title("FairFace Accuracy (higher=better)");axes[0,1].set_xticks(x);axes[0,1].set_xticklabels([a.capitalize() for a in attrs]);axes[0,1].legend()
# MaxSkew 3way
orig=[parse(v)[0] for v in retr["MS_OrigCLIP"]];ours=[parse(v)[0] for v in retr["MS_FairCLIP"]];zh=[parse(v)[0] for v in retr["MS_Zhang"]]
axes[1,0].bar(x-w3,orig,w3,label="Orig CLIP",color="#9aa0a6");axes[1,0].bar(x,zh,w3,label="Zhang",color="#ff7f0e");axes[1,0].bar(x+w3,ours,w3,label="FairCLIP",color="#1f77b4")
axes[1,0].set_title("FairFace MaxSkew (lower=fairer)");axes[1,0].set_xticks(x);axes[1,0].set_xticklabels([a.capitalize() for a in attrs]);axes[1,0].legend()
# NDKL 3way
orig=[parse(v)[0] for v in retr["NDKL_OrigCLIP"]];ours=[parse(v)[0] for v in retr["NDKL_FairCLIP"]];zh=[parse(v)[0] for v in retr["NDKL_Zhang"]]
axes[1,1].bar(x-w3,orig,w3,label="Orig CLIP",color="#9aa0a6");axes[1,1].bar(x,zh,w3,label="Zhang",color="#ff7f0e");axes[1,1].bar(x+w3,ours,w3,label="FairCLIP",color="#1f77b4")
axes[1,1].set_title("FairFace NDKL (lower=fairer)");axes[1,1].set_xticks(x);axes[1,1].set_xticklabels([a.capitalize() for a in attrs]);axes[1,1].legend()
fig.suptitle("FairCLIP — Complete Results Dashboard (FairFace, 3 seeds)",fontsize=15)
save(fig,"F_dashboard.png")

print("\nAll charts saved to", FIG)
