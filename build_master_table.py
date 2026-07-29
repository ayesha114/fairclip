"""Master results table.
Acc/F1/DPG  <- eval_all outputs (correct occupation-based DPG protocol)
EOD/RBS/MaxSkew/NDKL <- complete_metrics.py run
ABLE        <- recomputed from eval_all Acc + iterative MaxSkew
Zhang       <- published values (comparison only for Zhang's own metrics)
"""
import pandas as pd, numpy as np, os, re, csv

ZH={("ViT-B_32","gender"):(0.090,0.030,74.24),("ViT-B_32","age"):(0.572,0.364,59.60),("ViT-B_32","race"):(0.353,0.125,69.14),
    ("ViT-B_16","gender"):(0.080,0.025,78.35),("ViT-B_16","age"):(0.608,0.294,60.61),("ViT-B_16","race"):(0.353,0.125,69.14),
    ("ViT-L_14","gender"):(0.106,0.035,82.04),("ViT-L_14","age"):(0.579,0.332,64.25),("ViT-L_14","race"):(0.353,0.125,69.14),
    ("ViT-H_14","gender"):(0.138,0.051,82.11),("ViT-H_14","age"):(0.515,0.289,67.62),("ViT-H_14","race"):(0.353,0.125,69.14)}

def txt_metrics(path):
    if not os.path.exists(path): return None
    d={}
    for line in open(path, errors="ignore"):
        for k,p in [("Acc",r"Accuracy\s*:\s*([\d.]+)"),("F1",r"F1-Score\s*:\s*([\d.]+)"),("DPG",r"DPG\s*:\s*([\d.]+)")]:
            m=re.search(p,line)
            if m: d[k]=float(m.group(1))
    return d or None

def csv_metrics(path):
    if not os.path.exists(path): return None
    r=pd.read_csv(path).iloc[0]
    return {"Acc":r.get("accuracy"),"F1":r.get("f1"),"DPG":r.get("dpg")}

def get_cls(bbt, attr, method):
    if method=="Baseline":
        return txt_metrics(f"basefix_{bbt}_{attr}.txt") or csv_metrics(f"results/tables/eval_Baseline_CLIP_{bbt}_{attr}.csv")
    for p in [f"results/tables/eval_FairCLIP_{bbt}_{attr}_seed42.csv",
              f"bb_eval_{bbt}_{attr}.txt", f"h14_eval_{attr}.txt"]:
        r = csv_metrics(p) if p.endswith(".csv") else txt_metrics(p)
        if r: return r
    return None

cm = pd.read_csv("results/tables/complete_metrics_all.csv")
rows=[]
for bbt in ["ViT-B_32","ViT-B_16","ViT-L_14","ViT-H_14"]:
    for attr in ["gender","age","race"]:
        zms,znd,zable = ZH[(bbt,attr)]
        for method in ["Baseline","FairCLIP"]:
            cls = get_cls(bbt,attr,method)
            sub = cm[(cm.backbone==bbt)&(cm.attribute==attr)&(cm.method==method)]
            if cls is None or sub.empty: 
                print("missing:",bbt,attr,method); continue
            s = sub.iloc[0]
            acc = cls["Acc"]
            able = 2/(1/max(acc,1e-6)+1/np.exp(-s.MaxSkew))*100
            rows.append({"Backbone":bbt.replace("_","/"),"Attribute":attr,"Method":method,
                "Acc":round(acc,3),"F1":round(cls.get("F1",np.nan),3),"DPG":round(cls["DPG"],3),
                "EOD":round(s.EOD,3),"RBS":round(s.RBS,4),
                "MaxSkew":round(s.MaxSkew,3),"NDKL":round(s.NDKL,3),"ABLE":round(able,2)})
        rows.append({"Backbone":bbt.replace("_","/"),"Attribute":attr,"Method":"Zhang et al. (published)",
            "Acc":"N/R","F1":"N/R","DPG":"N/R","EOD":"N/R","RBS":"N/R",
            "MaxSkew":zms,"NDKL":znd,"ABLE":zable})

df=pd.DataFrame(rows)
df.to_csv("results/tables/publication/MASTER_all_metrics.csv", index=False)
pd.set_option("display.width",250,"display.max_columns",30)
for attr in ["gender","age","race"]:
    sub=df[df.Attribute==attr]
    print(f"\n{'='*110}\n  {attr.upper()}\n{'='*110}")
    print(sub.to_string(index=False))
    sub.to_csv(f"results/tables/publication/FINAL_{attr}.csv", index=False)

fc=df[df.Method=="FairCLIP"].copy()
wins_ms=wins_nd=tot=0
for _,r in fc.iterrows():
    bbt=r.Backbone.replace("/","_"); z=ZH[(bbt,r.Attribute)]
    tot+=1; wins_ms+= r.MaxSkew<z[0]; wins_nd+= r.NDKL<z[1]
print(f"\nFairCLIP beats Zhang: MaxSkew {wins_ms}/{tot} | NDKL {wins_nd}/{tot}")
print("saved results/tables/publication/MASTER_all_metrics.csv + FINAL_{gender,age,race}.csv")
