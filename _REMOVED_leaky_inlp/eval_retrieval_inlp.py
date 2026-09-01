"""
Retrieval-fairness evaluation with INLP debiasing (SEPARATE from classification).
Computes Zhang-comparable MaxSkew/NDKL for each attribute and compares to Zhang.
Usage: python eval_retrieval_inlp.py --attribute race
"""
import argparse, torch, numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from models.fairclip import FairCLIP
from models.inlp_retrieval_debias import fit_inlp_projection, apply_inlp
from data.datasets import FaceDataset, collate_dict
from evaluation.metrics.all_metrics import maxskew_ndkl_zhang, ZHANG_NEUTRAL_QUERIES

# Zhang published FairFace targets (ViT-B/32 for gender/age; ViT-B/16 race)
ZHANG = {
    "gender": {"orig_MS":0.138,"orig_NDKL":0.054,"MS":0.090,"NDKL":0.030},
    "age":    {"orig_MS":0.617,"orig_NDKL":0.416,"MS":0.572,"NDKL":0.364},
    "race":   {"orig_MS":0.528,"orig_NDKL":0.182,"MS":0.353,"NDKL":0.125},
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attribute", required=True, choices=["gender","age","race"])
    ap.add_argument("--dataset", default="fairface", choices=["fairface","utkface"])
    ap.add_argument("--backbone", default="ViT-B/32")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_iters", type=int, default=15)
    args = ap.parse_args()

    cfg = OmegaConf.load("configs/datasets/local_paths.yaml")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bb_tag = args.backbone.replace("/","_")

    m = FairCLIP(model_name=args.backbone, device=dev)
    # try dataset-folder path first (new layout), fall back to old flat path
    import os as _os
    p_new = f"results/checkpoints/{args.dataset}/{bb_tag}/{args.attribute}/seed{args.seed}/best_model.pt"
    p_old = f"results/checkpoints/{bb_tag}/{args.attribute}/seed{args.seed}/best_model.pt"
    ckpt = p_new if _os.path.exists(p_new) else p_old
    m.load(ckpt)
    m.eval()

    ds = FaceDataset(cfg[args.dataset].manifest, split="val", train=False)
    ebs = 32 if args.backbone in ("ViT-L/14","ViT-H/14") else 128
    ld = DataLoader(ds, batch_size=ebs, shuffle=False, collate_fn=collate_dict, num_workers=2)

    embs, labs = [], []
    with torch.no_grad():
        for b in ld:
            embs.append(m.encode_images(b["image"].to(dev)).cpu())
            labs.append(b[args.attribute].clone().detach())
    embs = torch.cat(embs).float(); labs = torch.cat(labs)
    embs = embs / embs.norm(dim=-1, keepdim=True)
    import gc; gc.collect(); torch.cuda.empty_cache()  # free image-encoder memory

    neutral = m.encode_text(ZHANG_NEUTRAL_QUERIES).cpu().float()
    neutral = neutral / neutral.norm(dim=-1, keepdim=True)

    # BEFORE INLP
    ms0, nd0 = maxskew_ndkl_zhang(neutral, embs, labs, k_skew=1000, k_ndkl=1000)

    # FIT + APPLY INLP
    P = fit_inlp_projection(embs, labs, max_iters=args.max_iters)
    embs_d = apply_inlp(embs, P)
    neutral_d = apply_inlp(neutral, P)
    ms1, nd1 = maxskew_ndkl_zhang(neutral_d, embs_d, labs, k_skew=1000, k_ndkl=1000)

    z = ZHANG[args.attribute]
    print(f"\n========== RETRIEVAL FAIRNESS: {args.attribute.upper()} ({args.backbone}) ==========")
    print(f"{'Method':<28}{'MaxSkew':>10}{'NDKL':>10}")
    print(f"{'Original CLIP (Zhang)':<28}{z['orig_MS']:>10.3f}{z['orig_NDKL']:>10.3f}")
    print(f"{'Zhang Ours (target)':<28}{z['MS']:>10.3f}{z['NDKL']:>10.3f}")
    print(f"{'FairCLIP (before INLP)':<28}{ms0:>10.3f}{nd0:>10.3f}")
    print(f"{'FairCLIP + INLP (ours)':<28}{ms1:>10.3f}{nd1:>10.3f}")
    beats_ms = "YES" if ms1 < z['MS'] else "no"
    beats_nd = "YES" if nd1 < z['NDKL'] else "no"
    print(f"\nBeats Zhang?  MaxSkew: {beats_ms}   NDKL: {beats_nd}")

    # save to CSV
    import csv, os
    os.makedirs("results/tables", exist_ok=True)
    _tag = "" if args.dataset=="fairface" else f"{args.dataset}_"
    out = f"results/tables/retrieval_inlp_{_tag}{args.attribute}_seed{args.seed}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["attribute","method","MaxSkew","NDKL"])
        w.writerow([args.attribute,"Original_CLIP_Zhang",z['orig_MS'],z['orig_NDKL']])
        w.writerow([args.attribute,"Zhang_Ours",z['MS'],z['NDKL']])
        w.writerow([args.attribute,"FairCLIP_before_INLP",round(ms0,3),round(nd0,3)])
        w.writerow([args.attribute,"FairCLIP_INLP",round(ms1,3),round(nd1,3)])
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
