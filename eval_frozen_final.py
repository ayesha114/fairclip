"""Baseline CLIP encoder + trained bias subspace (Step VIII). H/14-safe sequential."""
import argparse, torch
from models.fairclip import FairCLIP
from evaluation.retrieval import evaluate_retrieval
ap=argparse.ArgumentParser()
ap.add_argument("--backbone",default="ViT-B/32")
ap.add_argument("--model_path",required=True)
a=ap.parse_args()
t=FairCLIP(model_name=a.backbone,device="cuda")
t.load(a.model_path)
Bi=t.bias_discoverer.image_bias_directions
Bt=t.bias_discoverer.text_bias_directions
Bi=Bi.detach().cpu().clone() if Bi is not None else None
Bt=Bt.detach().cpu().clone() if Bt is not None else None
del t; torch.cuda.empty_cache()
m=FairCLIP(model_name=a.backbone,device="cuda")   # FROZEN baseline encoder
if Bi is not None:
    m.bias_remover.set_bias_subspace(Bi.cuda(),Bt.cuda())
    m._bias_subspace_fitted=True
m.eval()
with torch.no_grad():
    r=evaluate_retrieval(m,dataset="flickr30k",max_samples=1000,device="cuda")
print(f"[{a.backbone}] TR@5={r.get('TR@5',0):.1f} IR@5={r.get('IR@5',0):.1f}")
