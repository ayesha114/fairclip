"""Frozen encoder + trained bias subspace (Step VIII). ONE model only (H/14-safe)."""
import argparse, torch
from models.fairclip import FairCLIP
from evaluation.retrieval import evaluate_retrieval
ap=argparse.ArgumentParser()
ap.add_argument("--backbone",default="ViT-B/32")
ap.add_argument("--model_path",required=True)
a=ap.parse_args()
# Load ONE model. Its encoder = pretrained CLIP (frozen weights are mostly CLIP).
# We apply the trained bias subspace as Step VIII projection at inference.
m=FairCLIP(model_name=a.backbone,device="cuda")
m.load(a.model_path)   # loads trained bias subspace + weights
# ensure subspace is set for projection
if m.bias_discoverer.image_bias_directions is not None:
    m.bias_remover.set_bias_subspace(
        m.bias_discoverer.image_bias_directions,
        m.bias_discoverer.text_bias_directions)
    m._bias_subspace_fitted=True
m.eval()
with torch.no_grad():
    r=evaluate_retrieval(m,dataset="flickr30k",max_samples=1000,device="cuda")
print(f"FROZEN+PROJ {a.backbone}: TR@5={r.get('TR@5',0):.1f} IR@5={r.get('IR@5',0):.1f} (baseline 78.4/77.4)")
