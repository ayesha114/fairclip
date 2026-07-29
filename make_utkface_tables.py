"""UTKFace multi-seed results with mean±std vs Zhang (UTKFace targets)."""
import pandas as pd, os, numpy as np
TAB="results/tables"
attrs=["gender","age"]; seeds=[42,123,456]

# Zhang UTKFace-trained targets (Table 2, ViT-B/32, in-domain UTKFace)
ZHANG_UTK={"gender":{"MS":0.043,"NDKL":0.033,"orig_MS":0.066,"orig_NDKL":0.032},
           "age":   {"MS":0.407,"NDKL":0.252,"orig_MS":0.412,"orig_NDKL":0.253}}

def cls(attr,s):
    p=f"{TAB}/eval_FairCLIP_ViT-B_32_utkface_{attr}_seed{s}.csv"
    return pd.read_csv(p).iloc[0] if os.path.exists(p) else None
def retr(attr,s):
    p=f"{TAB}/retrieval_inlp_utkface_{attr}_seed{s}.csv"
    if os.path.exists(p):
        d=pd.read_csv(p); return d[d.method=="FairCLIP_INLP"].iloc[0]
    return None
def msd(vals):
    v=np.array([x for x in vals if x is not None],dtype=float)
    return f"{v.mean():.3f}±{v.std():.3f}" if len(v) else "—"

# Classification table
print("\n===== UTKFACE CLASSIFICATION (FairCLIP mean±std) =====")
crows=[]
for attr in attrs:
    accs=[cls(attr,s)["accuracy"] for s in seeds if cls(attr,s) is not None]
    f1s=[cls(attr,s)["f1"] for s in seeds if cls(attr,s) is not None]
    dpgs=[cls(attr,s)["dpg"] for s in seeds if cls(attr,s) is not None]
    crows.append({"Attribute":attr,"Acc":msd(accs),"F1":msd(f1s),"DPG":msd(dpgs)})
cdf=pd.DataFrame(crows)
pd.set_option("display.width",250,"display.max_columns",30)
print(cdf.to_string(index=False))
cdf.to_csv(f"{TAB}/UTKFACE_classification.csv",index=False)

# Retrieval table vs Zhang
print("\n===== UTKFACE RETRIEVAL: FairCLIP+INLP vs Zhang (UTKFace) =====")
rrows=[]
for attr in attrs:
    z=ZHANG_UTK[attr]
    ms=[retr(attr,s)["MaxSkew"] for s in seeds if retr(attr,s) is not None]
    nd=[retr(attr,s)["NDKL"] for s in seeds if retr(attr,s) is not None]
    rrows.append({"Attribute":attr,
        "MS_OrigCLIP":z["orig_MS"],"MS_FairCLIP":msd(ms),"MS_Zhang":z["MS"],
        "MS_beats":"YES" if np.mean(ms)<z["MS"] else "no",
        "NDKL_OrigCLIP":z["orig_NDKL"],"NDKL_FairCLIP":msd(nd),"NDKL_Zhang":z["NDKL"],
        "NDKL_beats":"YES" if np.mean(nd)<z["NDKL"] else "no"})
rdf=pd.DataFrame(rrows)
print(rdf.to_string(index=False))
rdf.to_csv(f"{TAB}/UTKFACE_retrieval.csv",index=False)
print(f"\nSaved: UTKFACE_classification.csv, UTKFACE_retrieval.csv")
