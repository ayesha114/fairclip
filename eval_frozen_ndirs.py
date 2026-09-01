"""Frozen encoder + trained subspace, with n_dirs cap. H/14-safe (one model)."""
import argparse, torch
from models.fairclip import FairCLIP
from evaluation.retrieval import evaluate_retrieval
ap=argparse.ArgumentParser()
ap.add_argument("--backbone",default="ViT-B/32")
ap.add_argument("--model_path",required=True)
ap.add_argument("--cap",type=int,default=None)  # limit number of bias directions
a=ap.parse_args()
m=FairCLIP(model_name=a.backbone,device="cuda")
m.load(a.model_path)
imgd=m.bias_discoverer.image_bias_directions
txtd=m.bias_discoverer.text_bias_directions
if imgd is not None and a.cap:
    imgd=imgd[:,:a.cap]; txtd=txtd[:,:a.cap]
m.bias_remover.set_bias_subspace(imgd,txtd)
m._bias_subspace_fitted=True
m.eval()
with torch.no_grad():
    r=evaluate_retrieval(m,dataset="flickr30k",max_samples=1000,device="cuda")
nd = imgd.shape[1] if imgd is not None else 0
print(f"FROZEN+PROJ {a.backbone} (n_dirs={nd}): TR@5={r.get('TR@5',0):.1f} IR@5={r.get('IR@5',0):.1f}")
