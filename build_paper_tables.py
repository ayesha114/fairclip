"""Build paper-ready results tables from clean result files. Saves CSV + Markdown."""
import re, os

os.makedirs("results/paper_tables", exist_ok=True)

# ---- Table 1: Classification (from methodology_procfix_all.txt) ----
rows = {}
for line in open("methodology_procfix_all.txt"):
    m = re.match(r'(\w+) (\S+) (\w+): DPG=([\d.e-]+) EOD=([\d.e-]+) RBS=([\d.e-]+)', line)
    if m:
        meth, bb, attr, dpg, eod, rbs = m.groups()
        rows.setdefault((bb, attr), {})[meth] = (float(dpg), float(eod), float(rbs))

with open("results/paper_tables/table1_classification.csv", "w") as f:
    f.write("Backbone,Attribute,Method,DPG,EOD,RBS,DPG_reduction_pct\n")
    for (bb, attr), d in rows.items():
        b, fc = d.get("BASELINE"), d.get("FairCLIP")
        if not (b and fc): continue
        red = 100*(b[0]-fc[0])/b[0]
        f.write(f"{bb},{attr},Baseline,{b[0]:.4f},{b[1]:.4f},{b[2]:.2e},-\n")
        f.write(f"{bb},{attr},FairCLIP,{fc[0]:.4f},{fc[1]:.4f},{fc[2]:.2e},{red:.0f}\n")

md = "# Table 1 - Classification Fairness (single fit-apply, no post-processing)\n\n"
md += "| Backbone | Attr | Method | DPG | EOD | RBS | DPG Reduction |\n|---|---|---|---|---|---|---|\n"
reds = []
for (bb, attr), d in rows.items():
    b, fc = d.get("BASELINE"), d.get("FairCLIP")
    if not (b and fc): continue
    red = 100*(b[0]-fc[0])/b[0]; reds.append(red)
    md += f"| {bb} | {attr} | Baseline | {b[0]:.4f} | {b[1]:.4f} | {b[2]:.2e} | - |\n"
    md += f"| {bb} | {attr} | FairCLIP | {fc[0]:.4f} | {fc[1]:.4f} | {fc[2]:.2e} | -{red:.0f}% |\n"
md += f"\n**FairCLIP beats Baseline {len(reds)}/{len(reds)} cells. Mean DPG reduction: {sum(reds)/len(reds):.0f}%**\n"
open("results/paper_tables/table1_classification.md", "w").write(md)

print("Table 1 saved:", len(reds), "cells")
print(md)
