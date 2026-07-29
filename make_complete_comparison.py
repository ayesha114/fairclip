"""Complete comparison: Baseline CLIP vs FairCLIP (mean±std) vs Zhang."""
import pandas as pd, os, numpy as np
TAB="results/tables"
attrs=["gender","age","race"]; seeds=[42,123,456]

# Zhang published (ViT-B/32 FairFace; race from B/16 ablation)
ZHANG={"gender":{"MS":0.090,"NDKL":0.030,"orig_MS":0.138,"orig_NDKL":0.054},
       "age":   {"MS":0.572,"NDKL":0.364,"orig_MS":0.617,"orig_NDKL":0.416},
       "race":  {"MS":0.353,"NDKL":0.125,"orig_MS":0.528,"orig_NDKL":0.182}}

def base(attr):
    p=f"{TAB}/eval_Baseline_CLIP_ViT-B_32_{attr}.csv"
    return pd.read_csv(p).iloc[0] if os.path.exists(p) else None
def cls(attr,s):
    p=f"{TAB}/eval_FairCLIP_ViT-B_32_{attr}_seed{s}.csv"
    return pd.read_csv(p).iloc[0] if os.path.exists(p) else None
def retr(attr,s):
    p=f"{TAB}/retrieval_inlp_{attr}_seed{s}.csv"
    if os.path.exists(p):
        d=pd.read_csv(p); return d[d.method=="FairCLIP_INLP"].iloc[0]
    return None

def msd(vals):
    v=np.array([x for x in vals if x is not None],dtype=float)
    return f"{v.mean():.3f}±{v.std():.3f}" if len(v) else "—"

# ---- CLASSIFICATION comparison (Baseline vs FairCLIP) ----
print("\n===== CLASSIFICATION: Baseline CLIP vs FairCLIP (mean±std) =====")
crows=[]
for attr in attrs:
    b=base(attr)
    accs=[cls(attr,s)["accuracy"] for s in seeds if cls(attr,s) is not None]
    dpgs=[cls(attr,s)["dpg"] for s in seeds if cls(attr,s) is not None]
    f1s=[cls(attr,s)["f1"] for s in seeds if cls(attr,s) is not None]
    crows.append({"Attribute":attr,
        "Acc_base":round(b["accuracy"],3),"Acc_FairCLIP":msd(accs),
        "F1_base":round(b["f1"],3),"F1_FairCLIP":msd(f1s),
        "DPG_base":round(b["dpg"],3),"DPG_FairCLIP":msd(dpgs)})
cdf=pd.DataFrame(crows)
pd.set_option("display.width",250,"display.max_columns",30)
print(cdf.to_string(index=False))
cdf.to_csv(f"{TAB}/COMPARE_classification.csv",index=False)

# ---- RETRIEVAL comparison (Original/Baseline vs FairCLIP vs Zhang) ----
print("\n===== RETRIEVAL: Original CLIP vs FairCLIP+INLP vs Zhang =====")
rrows=[]
for attr in attrs:
    z=ZHANG[attr]
    ms=[retr(attr,s)["MaxSkew"] for s in seeds if retr(attr,s) is not None]
    nd=[retr(attr,s)["NDKL"] for s in seeds if retr(attr,s) is not None]
    rrows.append({"Attribute":attr,
        "MS_OrigCLIP":z["orig_MS"],"MS_FairCLIP":msd(ms),"MS_Zhang":z["MS"],
        "MS_beats":"YES" if np.mean([x for x in ms])<z["MS"] else "no",
        "NDKL_OrigCLIP":z["orig_NDKL"],"NDKL_FairCLIP":msd(nd),"NDKL_Zhang":z["NDKL"],
        "NDKL_beats":"YES" if np.mean([x for x in nd])<z["NDKL"] else "no"})
rdf=pd.DataFrame(rrows)
print(rdf.to_string(index=False))
rdf.to_csv(f"{TAB}/COMPARE_retrieval.csv",index=False)
print(f"\nSaved: COMPARE_classification.csv, COMPARE_retrieval.csv")
