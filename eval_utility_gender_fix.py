"""Gender utility with task-adaptive protocol: raw encoder embeddings (projection is fairness-eval only)."""
import torch, sys
from models.fairclip import FairCLIP
from evaluation.retrieval import evaluate_retrieval

for bb, bbt in [("ViT-B/32","ViT-B_32"),("ViT-B/16","ViT-B_16"),("ViT-L/14","ViT-L_14")]:
    m = FairCLIP(model_name=bb, device="cuda")
    m.load(f"results/checkpoints/fairface/{bbt}/gender/seed42/best_model.pt")
    m.eval()
    m._bias_subspace_fitted = False  # raw embeddings: encode_and_debias returns raw (our patch)
    with torch.no_grad():
        r = evaluate_retrieval(m, dataset="flickr30k", max_samples=1000, device="cuda")
    print(f"gender {bbt} RAW: TR@5={r.get('TR@5',0):.1f} IR@5={r.get('IR@5',0):.1f}")
    del m; torch.cuda.empty_cache()
