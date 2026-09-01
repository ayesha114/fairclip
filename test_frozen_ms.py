import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from models.fairclip import FairCLIP
from data.datasets import FaceDataset, collate_dict
from evaluation.metrics.all_metrics import maxskew_ndkl_zhang, ZHANG_NEUTRAL_QUERIES
cfg=OmegaConf.load("configs/datasets/local_paths.yaml")
t=FairCLIP(model_name="ViT-B/32",device="cuda")
t.load("results/checkpoints_procfix/fairface/ViT-B_32/race/seed42/best_model.pt")
Bi=t.bias_discoverer.image_bias_directions.detach().cpu().float()
Bt=t.bias_discoverer.text_bias_directions.detach().cpu().float()
del t; torch.cuda.empty_cache()
m=FairCLIP(model_name="ViT-B/32",device="cuda"); m.eval()      # FROZEN baseline encoder
ds=FaceDataset(cfg.fairface.manifest,split="val",train=False)
ld=DataLoader(ds,batch_size=64,shuffle=False,collate_fn=collate_dict,num_workers=2)
E,L=[],[]
with torch.no_grad():
    for b in ld:
        E.append(m.backbone.encode_images(b["image"].to("cuda")).cpu()); L.append(b["race"].clone().detach())
E=torch.cat(E).float(); L=torch.cat(L); E=E/E.norm(dim=-1,keepdim=True)
neu=m.backbone.encode_text(ZHANG_NEUTRAL_QUERIES).cpu().float(); neu=neu/neu.norm(dim=-1,keepdim=True)
def prj(X,B):
    d=X-(X@B)@B.T
    return d/d.norm(dim=-1,keepdim=True).clamp(min=1e-8)
ms0,nd0=maxskew_ndkl_zhang(neu,E,L,k_skew=1000,k_ndkl=1000)
ms1,nd1=maxskew_ndkl_zhang(prj(neu,Bt),prj(E,Bi),L,k_skew=1000,k_ndkl=1000)
print(f"BASELINE encoder, no proj : MS={ms0:.3f} NDKL={nd0:.3f}")
print(f"BASELINE encoder + subspace: MS={ms1:.3f} NDKL={nd1:.3f}   (Zhang race 0.353/0.125)")
