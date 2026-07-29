"""
FairCLIP — Comparison Table Generator
Builds 3 tables: (1) FairCLIP vs Baseline (all metrics),
(2) FairCLIP vs Zhang (Zhang's metrics + beats?), (3) extra novelty (DPG/EOD/RBS).
Reads result CSVs from results/tables/.
"""
import glob, os
import pandas as pd

TAB = "results/tables"

# ---- Zhang FairFace published targets (the numbers to beat) ----
ZHANG = {
    "gender": {"MaxSkew": 0.080, "NDKL": 0.025, "ABLE": 78.35},
    "age":    {"MaxSkew": 0.608, "NDKL": 0.294, "ABLE": 60.61},
    "race":   {"MaxSkew": 0.353, "NDKL": 0.125, "ABLE": 69.14},
}

def load(method, attr):
    """Load a result row for method ('Baseline_CLIP' or 'FairCLIP') and attribute."""
    # try seed-specific first, then generic
    for pat in [f"{TAB}/eval_{method}_ViT-B_32_{attr}_seed42.csv",
                f"{TAB}/eval_{method}_ViT-B_32_{attr}.csv"]:
        if os.path.exists(pat):
            return pd.read_csv(pat).iloc[0]
    return None

rows_full, rows_zhang, rows_novel = [], [], []
for attr in ["gender", "age", "race"]:
    base = load("Baseline_CLIP", attr)
    fair = load("FairCLIP", attr)
    if base is None or fair is None:
        print(f"[skip] missing {attr} (base={base is not None}, fair={fair is not None})")
        continue

    # Table 1 — all metrics, baseline vs fairclip
    rows_full.append({
        "Attribute": attr, "Method": "Baseline CLIP",
        "Acc": base["accuracy"], "F1": base["f1"], "DPG": base["dpg"],
        "EOD": base["eod"], "RBS": base["rbs"], "MaxSkew": base["maxskew_5"],
        "NDKL": base["ndkl"], "ABLE": base["able"],
        "TR@5": base["TR@5"], "IR@5": base["IR@5"],
    })
    rows_full.append({
        "Attribute": attr, "Method": "FairCLIP (ours)",
        "Acc": fair["accuracy"], "F1": fair["f1"], "DPG": fair["dpg"],
        "EOD": fair["eod"], "RBS": fair["rbs"], "MaxSkew": fair["maxskew_5"],
        "NDKL": fair["ndkl"], "ABLE": fair["able"],
        "TR@5": fair["TR@5"], "IR@5": fair["IR@5"],
    })

    # Table 2 — FairCLIP vs Zhang (Zhang's metrics)
    z = ZHANG[attr]
    rows_zhang.append({
        "Attribute": attr,
        "MaxSkew (ours)": round(fair["maxskew_5"],3), "MaxSkew (Zhang)": z["MaxSkew"],
        "MaxSkew beats?": "YES" if fair["maxskew_5"] < z["MaxSkew"] else "no",
        "NDKL (ours)": round(fair["ndkl"],3), "NDKL (Zhang)": z["NDKL"],
        "NDKL beats?": "YES" if fair["ndkl"] < z["NDKL"] else "no",
        "ABLE (ours)": round(fair["able"],2), "ABLE (Zhang)": z["ABLE"],
        "ABLE beats?": "YES" if fair["able"] > z["ABLE"] else "no",
    })

    # Table 3 — extra novelty metrics Zhang does NOT report
    rows_novel.append({
        "Attribute": attr,
        "DPG base": base["dpg"], "DPG ours": fair["dpg"],
        "DPG reduction %": round(100*(base["dpg"]-fair["dpg"])/base["dpg"],1) if base["dpg"]>0 else 0,
        "EOD base": base["eod"], "EOD ours": fair["eod"],
        "RBS base": base["rbs"], "RBS ours": fair["rbs"],
    })

os.makedirs(TAB, exist_ok=True)

t1 = pd.DataFrame(rows_full)
t2 = pd.DataFrame(rows_zhang)
t3 = pd.DataFrame(rows_novel)

t1.to_csv(f"{TAB}/TABLE1_fairclip_vs_baseline.csv", index=False)
t2.to_csv(f"{TAB}/TABLE2_fairclip_vs_zhang.csv", index=False)
t3.to_csv(f"{TAB}/TABLE3_extra_novelty_dpg_eod_rbs.csv", index=False)

pd.set_option("display.width", 200, "display.max_columns", 30)
print("\n================ TABLE 1: FairCLIP vs Baseline CLIP (all metrics) ================")
print(t1.to_string(index=False))
print("\n================ TABLE 2: FairCLIP vs Zhang et al. (their metrics) ================")
print(t2.to_string(index=False))
print("\n================ TABLE 3: Extra novelty (DPG/EOD/RBS — Zhang does NOT report) ======")
print(t3.to_string(index=False))
print(f"\nSaved 3 CSVs to {TAB}/")
