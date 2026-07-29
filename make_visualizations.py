"""Generate thesis/paper figures from the final tables."""
import pandas as pd, numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAB = "results/tables"; FIG = "results/figures"; os.makedirs(FIG, exist_ok=True)
attrs = ["gender","age","race"]

t1 = pd.read_csv(f"{TAB}/FINAL_TABLE1_classification.csv")
t2 = pd.read_csv(f"{TAB}/FINAL_TABLE2_retrieval_vs_zhang.csv")

def save(fig,n): fig.savefig(f"{FIG}/{n}",dpi=150,bbox_inches="tight"); plt.close(fig); print("saved",n)

# FIG 1: DPG before/after (classification)
base = [t1[(t1.Attribute==a)&(t1.Method=="Baseline CLIP")]["DPG"].values[0] for a in attrs]
ours = [t1[(t1.Attribute==a)&(t1.Method=="FairCLIP (ours)")]["DPG"].values[0] for a in attrs]
x=np.arange(3); w=0.35
fig,ax=plt.subplots(figsize=(8,5))
ax.bar(x-w/2,base,w,label="Baseline CLIP",color="#9aa0a6")
ax.bar(x+w/2,ours,w,label="FairCLIP (ours)",color="#1f77b4")
for i,(b,o) in enumerate(zip(base,ours)):
    ax.text(i-w/2,b+.01,f"{b:.3f}",ha="center",fontsize=9)
    ax.text(i+w/2,o+.01,f"{o:.3f}",ha="center",fontsize=9)
ax.set_xticks(x);ax.set_xticklabels([a.capitalize() for a in attrs])
ax.set_ylabel("DPG (lower = fairer)");ax.set_title("Classification Fairness: DPG Before vs After FairCLIP")
ax.legend();save(fig,"fig1_dpg_classification.png")

# FIG 2: Accuracy before/after
base=[t1[(t1.Attribute==a)&(t1.Method=="Baseline CLIP")]["Acc"].values[0] for a in attrs]
ours=[t1[(t1.Attribute==a)&(t1.Method=="FairCLIP (ours)")]["Acc"].values[0] for a in attrs]
fig,ax=plt.subplots(figsize=(8,5))
ax.bar(x-w/2,base,w,label="Baseline CLIP",color="#9aa0a6")
ax.bar(x+w/2,ours,w,label="FairCLIP (ours)",color="#2ca02c")
for i,(b,o) in enumerate(zip(base,ours)):
    ax.text(i-w/2,b+.01,f"{b:.3f}",ha="center",fontsize=9)
    ax.text(i+w/2,o+.01,f"{o:.3f}",ha="center",fontsize=9)
ax.set_xticks(x);ax.set_xticklabels([a.capitalize() for a in attrs])
ax.set_ylabel("Accuracy (higher = better)");ax.set_title("Utility Preserved: Accuracy Before vs After FairCLIP")
ax.legend();save(fig,"fig2_accuracy.png")

# FIG 3: MaxSkew FairCLIP vs Zhang et al.
ms_ours=[t2[t2.Attribute==a]["MS_ours"].values[0] for a in attrs]
ms_zhang=[t2[t2.Attribute==a]["MS_Zhang"].values[0] for a in attrs]
fig,ax=plt.subplots(figsize=(8,5))
ax.bar(x-w/2,ms_zhang,w,label="Zhang et al. (CVPR2025)",color="#2ca02c")
ax.bar(x+w/2,ms_ours,w,label="FairCLIP (ours)",color="#1f77b4")
for i,(z,o) in enumerate(zip(ms_zhang,ms_ours)):
    ax.text(i-w/2,z+.005,f"{z:.3f}",ha="center",fontsize=9)
    ax.text(i+w/2,o+.005,f"{o:.3f}",ha="center",fontsize=9)
ax.set_xticks(x);ax.set_xticklabels([a.capitalize() for a in attrs])
ax.set_ylabel("MaxSkew (lower = fairer)");ax.set_title("Retrieval Fairness: Retrieval Fairness (MaxSkew): FairCLIP vs Zhang")
ax.legend();save(fig,"fig3_maxskew_vs_zhang.png")

# FIG 4: NDKL FairCLIP vs Zhang et al.
nd_ours=[t2[t2.Attribute==a]["NDKL_ours"].values[0] for a in attrs]
nd_zhang=[t2[t2.Attribute==a]["NDKL_Zhang"].values[0] for a in attrs]
fig,ax=plt.subplots(figsize=(8,5))
ax.bar(x-w/2,nd_zhang,w,label="Zhang et al.",color="#2ca02c")
ax.bar(x+w/2,nd_ours,w,label="FairCLIP (ours)",color="#1f77b4")
for i,(z,o) in enumerate(zip(nd_zhang,nd_ours)):
    ax.text(i-w/2,z+.005,f"{z:.3f}",ha="center",fontsize=9)
    ax.text(i+w/2,o+.005,f"{o:.3f}",ha="center",fontsize=9)
ax.set_xticks(x);ax.set_xticklabels([a.capitalize() for a in attrs])
ax.set_ylabel("NDKL (lower = fairer)");ax.set_title("Retrieval Fairness: Retrieval Fairness (NDKL): FairCLIP vs Zhang")
ax.legend();save(fig,"fig4_ndkl_vs_zhang.png")

print("All figures saved to", FIG)
