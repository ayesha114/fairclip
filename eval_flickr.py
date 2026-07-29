"""Flickr30k TR@5/IR@5 utility eval (baseline or FairCLIP). No INLP."""
import argparse, torch, csv, os
from models.fairclip import FairCLIP
from evaluation.retrieval import evaluate_retrieval

ap = argparse.ArgumentParser()
ap.add_argument("--model_path", default=None)
ap.add_argument("--backbone", default="ViT-B/32")
ap.add_argument("--baseline", action="store_true")
ap.add_argument("--tag", default="run")
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
m = FairCLIP(model_name=args.backbone, device=dev)
if not args.baseline and args.model_path:
    m.load(args.model_path)
m.eval()
if args.baseline:
    # force raw embeddings path (no debias) by hiding the method
    m.encode_and_debias_bak = m.encode_and_debias
    delattr(type(m), "encode_and_debias") if hasattr(type(m), "encode_and_debias") and False else None
with torch.no_grad():
    r = evaluate_retrieval(m, dataset="flickr30k", max_samples=1000, device=dev)
print(f"RESULT {args.tag}: TR@5={r.get('TR@5',0):.1f} IR@5={r.get('IR@5',0):.1f} "
      f"TR@1={r.get('TR@1',0):.1f} IR@1={r.get('IR@1',0):.1f}")
out = "results/tables/flickr_utility.csv"
new = not os.path.exists(out)
with open(out, "a", newline="") as f:
    w = csv.writer(f)
    if new: w.writerow(["tag","TR@1","TR@5","TR@10","IR@1","IR@5","IR@10"])
    w.writerow([args.tag]+[round(r.get(k,0),2) for k in ["TR@1","TR@5","TR@10","IR@1","IR@5","IR@10"]])
