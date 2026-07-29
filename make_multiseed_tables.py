"""Consolidated multi-seed results with mean±std (all metrics)."""
import pandas as pd, glob, os, numpy as np
TAB="results/tables"
attrs=["gender","age","race"]; seeds=[42,123,456]

ZHANG={"gender":{"MS":0.090,"NDKL":0.030},"age":{"MS":0.572,"NDKL":0.364},"race":{"MS":0.353,"NDKL":0.125}}

def load_cls(attr,seed):
    for p in [f"{TAB}/eval_FairCLIP_ViT-B_32_{attr}_seed{seed}.csv",
              f"{TAB}/eval_FairCLIP_ViT-B_32_{attr}.csv" if seed==42 else None]:
        if p and os.path.exists(p): return pd.read_csv(p).iloc[0]
    return None
def load_retr(attr,seed):
    p=f"{TAB}/retrieval_inlp_{attr}_seed{seed}.csv"
    if os.path.exists(p):
        df=pd.read_csv(p); return df[df.method=="FairCLIP_INLP"].iloc[0]
    return None

# ---- Per-seed full table ----
rows=[]
for attr in attrs:
    for seed in seeds:
        c=load_cls(attr,seed); r=load_retr(attr,seed)
        if c is None: continue
        rows.append({"Attribute":attr,"Seed":seed,
            "Acc":c["accuracy"],"F1":c["f1"],"DPG":c["dpg"],"EOD":c["eod"],
            "RBS":c["rbs"],"TR@5":c["TR@5"],"IR@5":c["IR@5"],
            "MaxSkew":r["MaxSkew"] if r is not None else None,
            "NDKL":r["NDKL"] if r is not None else None})
full=pd.DataFrame(rows)
full.to_csv(f"{TAB}/MULTISEED_full.csv",index=False)

# ---- Mean±std table ----
mrows=[]
for attr in attrs:
    sub=full[full.Attribute==attr]
    def ms(col): 
        v=sub[col].dropna().astype(float)
        return f"{v.mean():.3f} ± {v.std():.3f}" if len(v) else "—"
    mrows.append({"Attribute":attr,"Acc":ms("Acc"),"F1":ms("F1"),"DPG":ms("DPG"),
        "EOD":ms("EOD"),"RBS":ms("RBS"),"MaxSkew":ms("MaxSkew"),"NDKL":ms("NDKL"),
        "Zhang_MS":ZHANG[attr]["MS"],"Zhang_NDKL":ZHANG[attr]["NDKL"]})
mean=pd.DataFrame(mrows)
mean.to_csv(f"{TAB}/MULTISEED_mean_std.csv",index=False)

pd.set_option("display.width",250,"display.max_columns",30)
print("\n===== PER-SEED FULL METRICS (3 attributes x 3 seeds) =====")
print(full.to_string(index=False))
print("\n===== MEAN ± STD ACROSS SEEDS (vs Zhang) =====")
print(mean.to_string(index=False))
print(f"\nSaved: MULTISEED_full.csv, MULTISEED_mean_std.csv")
