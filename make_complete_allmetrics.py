"""Complete all-metrics comparison: Baseline vs FairCLIP vs Zhang (both backbones)."""
import pandas as pd, os, numpy as np
TAB="results/tables"

# Zhang published (MS/NDKL/ABLE). DPG/EOD/RBS = Zhang did NOT report (N/R).
ZHANG={
 "ViT-B_32":{"gender":{"MS":0.090,"NDKL":0.030,"oMS":0.138,"oNDKL":0.054},
             "age":{"MS":0.572,"NDKL":0.364,"oMS":0.617,"oNDKL":0.416},
             "race":{"MS":0.353,"NDKL":0.125,"oMS":0.528,"oNDKL":0.182}},
 "ViT-B_16":{"gender":{"MS":0.080,"NDKL":0.025,"oMS":0.218,"oNDKL":0.088},
             "age":{"MS":0.608,"NDKL":0.294,"oMS":0.657,"oNDKL":0.433},
             "race":{"MS":0.353,"NDKL":0.125,"oMS":0.528,"oNDKL":0.182}},
}

def base(bb,attr):
    p=f"{TAB}/eval_Baseline_CLIP_{bb}_{attr}.csv"
    return pd.read_csv(p).iloc[0] if os.path.exists(p) else None
def cls(bb,attr,s):
    # try seed-tagged, then generic
    for p in [f"{TAB}/eval_FairCLIP_{bb}_{attr}_seed{s}.csv"]:
        if os.path.exists(p): return pd.read_csv(p).iloc[0]
    return None
def retr(bb,attr,s):
    tag = "" if bb=="ViT-B_32" else "vitb16_"
    p=f"{TAB}/retrieval_inlp_{tag}{attr}_seed{s}.csv"
    # debug-safe: also try alternate
    if not os.path.exists(p) and bb=="ViT-B_32":
        p=f"{TAB}/retrieval_inlp_{attr}_seed{s}.csv"
    if os.path.exists(p):
        d=pd.read_csv(p); 
        r=d[d.method=="FairCLIP_INLP"]
        return r.iloc[0] if len(r) else None
    return None
def msd(vals):
    v=np.array([x for x in vals if x is not None],dtype=float)
    return f"{v.mean():.3f}±{v.std():.3f}" if len(v) else "—"

seeds=[42,123,456]
for bb in ["ViT-B_32","ViT-B_16"]:
    print(f"\n{'='*70}\n  BACKBONE: {bb}  —  COMPLETE METRIC COMPARISON\n{'='*70}")
    rows=[]
    for attr in ["gender","age","race"]:
        b=base(bb,attr)
        if b is None: 
            print(f"  [no baseline for {attr}]"); continue
        accs=[cls(bb,attr,s)["accuracy"] for s in seeds if cls(bb,attr,s) is not None]
        f1s =[cls(bb,attr,s)["f1"] for s in seeds if cls(bb,attr,s) is not None]
        dpgs=[cls(bb,attr,s)["dpg"] for s in seeds if cls(bb,attr,s) is not None]
        eods=[cls(bb,attr,s)["eod"] for s in seeds if cls(bb,attr,s) is not None]
        rbss=[cls(bb,attr,s)["rbs"] for s in seeds if cls(bb,attr,s) is not None]
        mss =[retr(bb,attr,s)["MaxSkew"] for s in seeds if retr(bb,attr,s) is not None]
        nds =[retr(bb,attr,s)["NDKL"] for s in seeds if retr(bb,attr,s) is not None]
        z=ZHANG[bb][attr]
        rows.append({"Attr":attr,
          "Acc_base":round(b['accuracy'],3),"Acc_ours":msd(accs),
          "F1_base":round(b['f1'],3),"F1_ours":msd(f1s),
          "DPG_base":round(b['dpg'],3),"DPG_ours":msd(dpgs),"DPG_Zhang":"N/R",
          "EOD_base":round(b['eod'],3),"EOD_ours":msd(eods),"EOD_Zhang":"N/R",
          "RBS_base":round(b['rbs'],3),"RBS_ours":msd(rbss),"RBS_Zhang":"N/R",
          "MS_orig":z['oMS'],"MS_ours":msd(mss),"MS_Zhang":z['MS'],
          "NDKL_orig":z['oNDKL'],"NDKL_ours":msd(nds),"NDKL_Zhang":z['NDKL']})
    df=pd.DataFrame(rows)
    pd.set_option("display.width",300,"display.max_columns",40)
    # print in 3 blocks for readability
    print("\n  [Classification: Accuracy & F1]")
    print(df[["Attr","Acc_base","Acc_ours","F1_base","F1_ours"]].to_string(index=False))
    print("\n  [Classification Fairness: DPG / EOD / RBS — Zhang did Not Report (N/R)]")
    print(df[["Attr","DPG_base","DPG_ours","DPG_Zhang","EOD_base","EOD_ours","RBS_base","RBS_ours"]].to_string(index=False))
    print("\n  [Retrieval Fairness: MaxSkew & NDKL — OrigCLIP / Ours / Zhang]")
    print(df[["Attr","MS_orig","MS_ours","MS_Zhang","NDKL_orig","NDKL_ours","NDKL_Zhang"]].to_string(index=False))
    df.to_csv(f"{TAB}/ALLMETRICS_{bb}.csv",index=False)
    print(f"\n  Saved: ALLMETRICS_{bb}.csv")
