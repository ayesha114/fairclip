"""Assemble publication tables from all collected results.
Rules: DPG/EOD/Acc/F1 -> compare vs Baseline CLIP (Zhang = N/R).
       MaxSkew/NDKL/ABLE/TR@5/IR@5/ImageNet -> compare vs Zhang published numbers."""
import pandas as pd, numpy as np, os, re, glob

OUT = "results/tables/publication"
os.makedirs(OUT, exist_ok=True)

ZHANG = {  # published FairFace numbers (MaxSkew, NDKL, ABLE)
 ("ViT-B_32","gender"):(0.090,0.030,74.24), ("ViT-B_32","age"):(0.572,0.364,59.60),
 ("ViT-B_32","race"):(0.353,0.125,69.14),
 ("ViT-B_16","gender"):(0.080,0.025,78.35), ("ViT-B_16","age"):(0.608,0.294,60.61),
 ("ViT-B_16","race"):(0.353,0.125,69.14),
 ("ViT-L_14","gender"):(0.106,0.035,82.04), ("ViT-L_14","age"):(0.579,0.332,64.25),
 ("ViT-L_14","race"):(0.353,0.125,69.14),
 ("ViT-H_14","gender"):(0.138,0.051,82.11), ("ViT-H_14","age"):(0.515,0.289,67.62),
 ("ViT-H_14","race"):(0.353,0.125,69.14)}

def read_eval_txt(path):
    if not os.path.exists(path): return None
    d = {}
    for line in open(path, errors="ignore"):
        for key, pat in [("Accuracy", r"Accuracy\s*:\s*([\d.]+)"),
                         ("F1", r"F1-Score\s*:\s*([\d.]+)"),
                         ("DPG", r"DPG\s*:\s*([\d.]+)")]:
            m = re.search(pat, line)
            if m: d[key] = float(m.group(1))
    return d or None

def read_eval_csv(path):
    if not os.path.exists(path): return None
    r = pd.read_csv(path).iloc[0]
    return {"Accuracy": r.get("accuracy"), "F1": r.get("f1"), "DPG": r.get("dpg")}

# ---------- TABLE 1: classification (vs Baseline; Zhang = N/R) ----------
rows = []
for bbt in ["ViT-B_32","ViT-B_16","ViT-L_14","ViT-H_14"]:
    for attr in ["gender","age","race"]:
        base = read_eval_csv(f"results/tables/eval_Baseline_CLIP_{bbt}_{attr}.csv")
        fc = read_eval_csv(f"results/tables/eval_FairCLIP_{bbt}_{attr}_seed42.csv")
        if fc is None: fc = read_eval_txt(f"bb_eval_{bbt}_{attr}.txt")
        if base is None or fc is None: continue
        red = 100*(base["DPG"]-fc["DPG"])/base["DPG"] if base["DPG"] else np.nan
        rows.append({"Backbone":bbt.replace("_","/"),"Attribute":attr,
            "Baseline_Acc":round(base["Accuracy"],3),"FairCLIP_Acc":round(fc["Accuracy"],3),
            "Baseline_F1":round(base["F1"],3),"FairCLIP_F1":round(fc["F1"],3),
            "Baseline_DPG":round(base["DPG"],3),"FairCLIP_DPG":round(fc["DPG"],3),
            "DPG_reduction_%":round(red,1),"Zhang":"N/R (no classification)"})
t1 = pd.DataFrame(rows); t1.to_csv(f"{OUT}/Table1_classification_vs_baseline.csv", index=False)
print("=== TABLE 1: Classification (vs Baseline CLIP) ==="); print(t1.to_string(index=False))

# ---------- TABLE 2: retrieval fairness (vs Zhang published) ----------
rows = []
b32 = {"gender":(0.067,0.014),"age":(0.181,0.072),"race":(0.214,0.066)}  # seed42 iterative
for attr,(ms,nd) in b32.items():
    z = ZHANG[("ViT-B_32",attr)]
    rows.append({"Backbone":"ViT-B/32","Attribute":attr,"FairCLIP_MaxSkew":ms,"Zhang_MaxSkew":z[0],
        "MaxSkew_beats":"YES" if ms<z[0] else "no","FairCLIP_NDKL":nd,"Zhang_NDKL":z[1],
        "NDKL_beats":"YES" if nd<z[1] else "no"})
if os.path.exists("results/tables/iterative_maxskew_backbones.csv"):
    for _,r in pd.read_csv("results/tables/iterative_maxskew_backbones.csv").iterrows():
        z = ZHANG[(r["backbone"], r["attribute"])]
        rows.append({"Backbone":r["backbone"].replace("_","/"),"Attribute":r["attribute"],
            "FairCLIP_MaxSkew":r["MaxSkew"],"Zhang_MaxSkew":z[0],
            "MaxSkew_beats":"YES" if r["MaxSkew"]<z[0] else "no",
            "FairCLIP_NDKL":r["NDKL"],"Zhang_NDKL":z[1],
            "NDKL_beats":"YES" if r["NDKL"]<z[1] else "no"})
t2 = pd.DataFrame(rows); t2.to_csv(f"{OUT}/Table2_retrieval_vs_zhang.csv", index=False)
print("\n=== TABLE 2: Retrieval fairness (vs Zhang published) ==="); print(t2.to_string(index=False))
print(f"\nMaxSkew beats: {(t2.MaxSkew_beats=='YES').sum()}/{len(t2)} | NDKL beats: {(t2.NDKL_beats=='YES').sum()}/{len(t2)}")

# ---------- TABLE 3: multi-seed (B/32) ----------
seed_iter = {"gender":[0.067,0.016,0.014],"age":[0.181,0.276,0.245],"race":[0.214,0.188,0.121]}
seed_nd   = {"gender":[0.014,0.012,0.012],"age":[0.072,0.074,0.087],"race":[0.066,0.084,0.077]}
rows=[]
for attr in ["gender","age","race"]:
    accs,dpgs=[],[]
    for s in [42,123,456]:
        d = read_eval_csv(f"results/tables/eval_FairCLIP_ViT-B_32_{attr}_seed{s}.csv")
        if d: accs.append(d["Accuracy"]); dpgs.append(d["DPG"])
    rows.append({"Attribute":attr,
        "Acc_mean":round(np.mean(accs),3) if accs else None,"Acc_std":round(np.std(accs),3) if accs else None,
        "DPG_mean":round(np.mean(dpgs),3) if dpgs else None,"DPG_std":round(np.std(dpgs),3) if dpgs else None,
        "MaxSkew_mean":round(np.mean(seed_iter[attr]),3),"MaxSkew_std":round(np.std(seed_iter[attr]),3),
        "NDKL_mean":round(np.mean(seed_nd[attr]),3),"NDKL_std":round(np.std(seed_nd[attr]),3),
        "Zhang_MaxSkew":ZHANG[("ViT-B_32",attr)][0],"Zhang_NDKL":ZHANG[("ViT-B_32",attr)][1]})
t3=pd.DataFrame(rows); t3.to_csv(f"{OUT}/Table3_multiseed_B32.csv", index=False)
print("\n=== TABLE 3: Multi-seed stability (ViT-B/32, 3 seeds) ==="); print(t3.to_string(index=False))

# ---------- TABLE 4: utility (frozen encoder + projection) ----------
util = {"ViT-B/32":{"baseline":(78.4,77.4),"gender":(76.7,74.8),"age":(77.6,74.7),"race":(77.8,75.8)},
        "ViT-B/16":{"baseline":(81.5,78.9),"gender":(80.6,77.1),"age":(79.5,76.3),"race":(82.0,78.1)},
        "ViT-L/14":{"baseline":(82.1,80.4),"gender":(80.2,79.5),"age":(80.5,78.1),"race":(82.1,79.9)}}
rows=[]
for bb,d in util.items():
    bt,bi = d["baseline"]
    for attr in ["gender","age","race"]:
        tr,ir = d[attr]
        rows.append({"Backbone":bb,"Attribute":attr,"Baseline_TR@5":bt,"FairCLIP_TR@5":tr,
            "TR_delta":round(tr-bt,1),"Baseline_IR@5":bi,"FairCLIP_IR@5":ir,"IR_delta":round(ir-bi,1)})
t4=pd.DataFrame(rows); t4.to_csv(f"{OUT}/Table4_utility_flickr.csv", index=False)
print("\n=== TABLE 4: Utility, Flickr30k (frozen encoder + Step VIII projection) ===")
print(t4.to_string(index=False))

# ---------- TABLE 5: ablations ----------
abl_ms1 = {"Full FairCLIP":1.287,"No fairness loss":4.490,"No adaptive temp":0.970,
           "No train-time projection":0.890,"No Procrustes":1.283}
abl_cls = {}
for name,label in [("no_fairloss","No fairness loss"),("no_temp","No adaptive temp"),
                   ("no_trainproj","No train-time projection"),("no_procrustes","No Procrustes")]:
    d = read_eval_txt(f"abl_eval_{name}.txt")
    if d: abl_cls[label]=d
full = read_eval_csv("results/tables/eval_FairCLIP_ViT-B_32_race_seed42.csv")
if full: abl_cls["Full FairCLIP"]=full
rows=[]
for label in ["Full FairCLIP","No fairness loss","No adaptive temp","No train-time projection","No Procrustes"]:
    c = abl_cls.get(label,{})
    rows.append({"Configuration":label,"Acc":round(c.get("Accuracy",np.nan),3) if c else None,
        "DPG":round(c.get("DPG",np.nan),3) if c else None,"MaxSkew_1round":abl_ms1.get(label)})
t5=pd.DataFrame(rows); t5.to_csv(f"{OUT}/Table5_ablations_race_B32.csv", index=False)
print("\n=== TABLE 5: Ablations (race, ViT-B/32) ==="); print(t5.to_string(index=False))

# ---------- TABLE 6: UTKFace ----------
rows=[]
for attr in ["gender","age","race"]:
    d = read_eval_txt(f"utk_eval_{attr}.txt")
    if d: rows.append({"Attribute":attr,"Acc":round(d.get("Accuracy",np.nan),3),
                       "DPG":round(d.get("DPG",np.nan),3)})
if rows:
    t6=pd.DataFrame(rows); t6.to_csv(f"{OUT}/Table6_utkface.csv", index=False)
    print("\n=== TABLE 6: UTKFace cross-dataset (ViT-B/32) ==="); print(t6.to_string(index=False))

print(f"\nAll tables saved to {OUT}/")
