"""
Final comparison tables for thesis/paper.
TABLE 1: Classification fairness — FairCLIP vs Baseline (DPG/EOD/RBS/Acc/F1)
TABLE 2: Retrieval fairness — FairCLIP+INLP vs Zhang (MaxSkew/NDKL)
TABLE 3: DPG/EOD/RBS — extra novelty (Zhang does not report these)
"""
import pandas as pd, glob, os
TAB = "results/tables"

# Zhang published FairFace ViT-B/32 (race from ViT-B/16 ablation)
ZHANG = {
    "gender": {"MS":0.090,"NDKL":0.030,"ABLE":74.24},
    "age":    {"MS":0.572,"NDKL":0.364,"ABLE":59.60},
    "race":   {"MS":0.353,"NDKL":0.125,"ABLE":69.14},
}

def load(method, attr):
    for p in [f"{TAB}/eval_{method}_ViT-B_32_{attr}_seed42.csv",
              f"{TAB}/eval_{method}_ViT-B_32_{attr}.csv"]:
        if os.path.exists(p): return pd.read_csv(p).iloc[0]
    return None

def load_retr(attr):
    p = f"{TAB}/retrieval_inlp_{attr}_seed42.csv"
    if os.path.exists(p):
        df = pd.read_csv(p)
        return df[df.method=="FairCLIP_INLP"].iloc[0]
    return None

t1, t2, t3 = [], [], []
for attr in ["gender","age","race"]:
    base, fair, retr = load("Baseline_CLIP",attr), load("FairCLIP",attr), load_retr(attr)
    if base is None or fair is None: 
        print(f"[skip classification] {attr}"); continue

    # TABLE 1 — classification fairness
    for nm, row in [("Baseline CLIP",base),("FairCLIP (ours)",fair)]:
        t1.append({"Attribute":attr,"Method":nm,"Acc":row["accuracy"],"F1":row["f1"],
                   "DPG":row["dpg"],"EOD":row["eod"],"RBS":row["rbs"]})

    # TABLE 3 — DPG/EOD/RBS novelty
    red = round(100*(base["dpg"]-fair["dpg"])/base["dpg"],1) if base["dpg"]>0 else 0
    t3.append({"Attribute":attr,"DPG_base":base["dpg"],"DPG_ours":fair["dpg"],
               "DPG_reduction_%":red,"EOD_base":base["eod"],"EOD_ours":fair["eod"],
               "RBS_base":base["rbs"],"RBS_ours":fair["rbs"]})

    # TABLE 2 — retrieval fairness vs Zhang
    if retr is not None:
        z = ZHANG[attr]
        t2.append({"Attribute":attr,
                   "MS_ours":retr["MaxSkew"],"MS_Zhang":z["MS"],
                   "MS_beats":"YES" if retr["MaxSkew"]<z["MS"] else "no",
                   "NDKL_ours":retr["NDKL"],"NDKL_Zhang":z["NDKL"],
                   "NDKL_beats":"YES" if retr["NDKL"]<z["NDKL"] else "no"})

T1, T2, T3 = pd.DataFrame(t1), pd.DataFrame(t2), pd.DataFrame(t3)
T1.to_csv(f"{TAB}/FINAL_TABLE1_classification.csv", index=False)
T2.to_csv(f"{TAB}/FINAL_TABLE2_retrieval_vs_zhang.csv", index=False)
T3.to_csv(f"{TAB}/FINAL_TABLE3_dpg_eod_rbs.csv", index=False)

pd.set_option("display.width",200,"display.max_columns",30)
print("\n===== TABLE 1: CLASSIFICATION FAIRNESS (FairCLIP vs Baseline) =====")
print(T1.to_string(index=False))
print("\n===== TABLE 2: RETRIEVAL FAIRNESS (FairCLIP+INLP vs Zhang) =====")
print(T2.to_string(index=False))
print("\n===== TABLE 3: DPG/EOD/RBS — extra novelty (Zhang omits) =====")
print(T3.to_string(index=False))
print(f"\nSaved 3 CSVs to {TAB}/")
