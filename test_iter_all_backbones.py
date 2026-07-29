"""Iterative Step IV+VIII eval for B/16 and L/14, all attributes."""
import torch, csv
from omegaconf import OmegaConf
from models.fairclip import FairCLIP
from data.datasets import FaceDataset, collate_dict
from torch.utils.data import DataLoader
from evaluation.metrics.all_metrics import maxskew_ndkl_zhang, ZHANG_NEUTRAL_QUERIES

cfg = OmegaConf.load("configs/datasets/local_paths.yaml")
K = {"gender": 2, "race": 6, "age": 8}
rows = []
for bb, bbt in [("ViT-B/16","ViT-B_16"), ("ViT-L/14","ViT-L_14")]:
    for attr in ["gender","age","race"]:
        import os
        ck = f"results/checkpoints/fairface/{bbt}/{attr}/seed42/best_model.pt"
        if not os.path.exists(ck): print("missing", ck); continue
        m = FairCLIP(model_name=bb, device="cuda"); m.load(ck); m.eval()
        ds = FaceDataset(cfg.fairface.manifest, split="val", train=False)
        ld = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_dict, num_workers=2)
        img, labs = [], []
        with torch.no_grad():
            for b in ld:
                img.append(m.backbone.encode_images(b["image"].to("cuda")).cpu())
                labs.append(b[attr].clone().detach())
        img = torch.cat(img).float(); labs = torch.cat(labs)
        img = img/img.norm(dim=-1, keepdim=True)
        txt = m.encode_text(ZHANG_NEUTRAL_QUERIES).cpu().float()
        txt = txt/txt.norm(dim=-1, keepdim=True)
        def fit(E, labels, k):
            g = sorted(labels.unique().tolist())
            c = torch.stack([E[labels==x].mean(0) for x in g])
            _,_,Vt = torch.linalg.svd(c - c.mean(0, keepdim=True), full_matrices=False)
            return Vt.T[:, :k]
        def proj(E, B):
            d = E - (E@B)@B.T
            return d/d.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        ei, et = img.clone(), txt.clone()
        best = (9e9, 9e9, 0)
        for r in range(1, 7):
            B = fit(ei, labs, K[attr])
            ei = proj(ei, B); et = proj(et, B)
            ms, nd = maxskew_ndkl_zhang(et, ei, labs, k_skew=1000, k_ndkl=1000)
            if ms < best[0]: best = (ms, nd, r)
        print(f"{bbt} {attr}: best MaxSkew={best[0]:.3f} NDKL={best[1]:.3f} (round {best[2]})")
        rows.append([bbt, attr, round(best[0],3), round(best[1],3), best[2]])
        del m; torch.cuda.empty_cache()
with open("results/tables/iterative_maxskew_backbones.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["backbone","attribute","MaxSkew","NDKL","round"]); w.writerows(rows)
print("saved results/tables/iterative_maxskew_backbones.csv")
