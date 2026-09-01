"""Build clean paper-ready tables from verified result files. No re-eval."""
import re, csv
from collections import defaultdict

seed42 = {}
for line in open("final_results.txt"):
    m = re.search(r'\[(ViT-[\w/]+) (\w+)\]\s*(?:DPG=([\d.]+) EOD=([\d.]+) RBS=([\d.e-]+))?\s*\|?\s*(?:TR@5=([\d.]+) IR@5=([\d.]+))?', line)
    if not m: continue
    bb, attr = m.group(1), m.group(2)
    key = (bb, attr)
    if key not in seed42: seed42[key] = {}
    if m.group(3): seed42[key].update(DPG=m.group(3), EOD=m.group(4), RBS=m.group(5))
    if m.group(6): seed42[key].update(TR5=m.group(6), IR5=m.group(7))

ms = {}
try:
    r = csv.DictReader(open("results/multiseed_meanstd.txt"))
    for row in r:
        ms[row["Cell"]] = row
except: pass

comp = defaultdict(dict)
for line in open("complete_metrics.txt"):
    m = re.search(r'(ViT-[\w_]+) (\w+) (Baseline|FairCLIP):\s*Acc=([\d.]+).*MaxSkew=([\d.]+) NDKL=([\d.]+) ABLE=([\d.]+)', line)
    if not m: continue
    bb = m.group(1).replace("_","/"); attr=m.group(2); which=m.group(3)
    comp[(bb,attr)][which] = dict(Acc=m.group(4), MaxSkew=m.group(5), NDKL=m.group(6), ABLE=m.group(7))

ZHANG = {
    ("gender","MaxSkew"):"0.080", ("gender","NDKL"):"0.025", ("gender","ABLE"):"78.35",
    ("age","MaxSkew"):"0.608", ("age","NDKL"):"0.294", ("age","ABLE"):"60.61",
    ("race","MaxSkew"):"0.353", ("race","NDKL"):"0.125", ("race","ABLE"):"69.14",
}
backbones=["ViT-B/32","ViT-B/16","ViT-L/14","ViT-H/14"]
attrs=["gender","age","race"]

with open("results/TABLE1_main.csv","w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["Backbone","Attribute","Acc_Base","Acc_FairCLIP",
        "MaxSkew_Base","MaxSkew_FairCLIP","MaxSkew_Zhang",
        "NDKL_Base","NDKL_FairCLIP","NDKL_Zhang",
        "ABLE_Base","ABLE_FairCLIP","ABLE_Zhang"])
    for bb in backbones:
        for attr in attrs:
            c=comp.get((bb,attr),{}); b=c.get("Baseline",{}); fc=c.get("FairCLIP",{})
            w.writerow([bb,attr,b.get("Acc",""),fc.get("Acc",""),
                b.get("MaxSkew",""),fc.get("MaxSkew",""),ZHANG.get((attr,"MaxSkew"),""),
                b.get("NDKL",""),fc.get("NDKL",""),ZHANG.get((attr,"NDKL"),""),
                b.get("ABLE",""),fc.get("ABLE",""),ZHANG.get((attr,"ABLE"),"")])

with open("results/TABLE2_fairness.csv","w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["Backbone","Attribute","DPG_seed42","EOD_seed42","RBS_seed42",
                "DPG_mean","DPG_std","EOD_mean","EOD_std"])
    for bb in backbones:
        for attr in attrs:
            s=seed42.get((bb,attr),{}); mm=ms.get(f"{bb} {attr}",{})
            w.writerow([bb,attr,s.get("DPG",""),s.get("EOD",""),s.get("RBS",""),
                mm.get("DPG_mean",""),mm.get("DPG_std",""),mm.get("EOD_mean",""),mm.get("EOD_std","")])

with open("results/TABLE3_retrieval.csv","w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["Backbone","Attribute","TR@5","IR@5","Baseline_TR@5"])
    base_tr={"ViT-B/32":"78.4","ViT-B/16":"~85","ViT-L/14":"~86","ViT-H/14":"89.1"}
    for bb in backbones:
        for attr in attrs:
            s=seed42.get((bb,attr),{})
            w.writerow([bb,attr,s.get("TR5",""),s.get("IR5",""),base_tr.get(bb,"")])

print("WROTE 3 tables")
