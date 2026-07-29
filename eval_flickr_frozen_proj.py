"""Utility with frozen pretrained encoder + trained projection (Step VIII) only."""
import torch
from models.fairclip import FairCLIP
from evaluation.retrieval import evaluate_retrieval

for attr in ["gender","age","race"]:
    trained = FairCLIP(model_name="ViT-B/32", device="cuda")
    trained.load(f"results/checkpoints/fairface/ViT-B_32/{attr}/seed42/best_model.pt")
    fresh = FairCLIP(model_name="ViT-B/32", device="cuda")  # frozen pretrained encoder
    # transplant trained bias subspace onto frozen encoder
    fresh.bias_discoverer.image_bias_directions = trained.bias_discoverer.image_bias_directions
    fresh.bias_discoverer.text_bias_directions = trained.bias_discoverer.text_bias_directions
    fresh.bias_remover.set_bias_subspace(
        trained.bias_discoverer.image_bias_directions,
        trained.bias_discoverer.text_bias_directions)
    fresh._bias_subspace_fitted = True
    fresh.eval()
    with torch.no_grad():
        r = evaluate_retrieval(fresh, dataset="flickr30k", max_samples=1000, device="cuda")
    print(f"frozen+proj {attr}: TR@5={r.get('TR@5',0):.1f} IR@5={r.get('IR@5',0):.1f}  (baseline 78.4/77.4)")
    del trained, fresh; torch.cuda.empty_cache()
