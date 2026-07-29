#!/usr/bin/env bash
cd /home/ayeshakalsoom/Downloads/thesis
source /home/ayeshakalsoom/Downloads/fairclip-thesis/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# FIX A: L/14 gender NDKL - more bias directions, smaller lambda
rm -rf results/checkpoints_fix/ViT-L_14/gender/seed42
python -m training.train --dataset fairface --attribute gender --backbone ViT-L/14 \
  --epochs 30 --patience 3 --batch_size 16 --lambda_fair 0.005 --n_bias_dirs 8 --seed 42 \
  --output_dir results/checkpoints_fix > fixA_l14_gender.txt 2>&1

# FIX B: H/14 age accuracy - smaller lambda
rm -rf results/checkpoints_fix/ViT-H_14/age/seed42
python -m training.train --dataset fairface --attribute age --backbone ViT-H/14 \
  --epochs 30 --patience 3 --batch_size 8 --lambda_fair 0.25 --n_bias_dirs 5 --seed 42 \
  --output_dir results/checkpoints_fix > fixB_h14_age.txt 2>&1

# evaluate both
for spec in "ViT-L/14|ViT-L_14|gender|1,2,3|12" "ViT-H/14|ViT-H_14|age|6,8|8"; do
  BB=$(echo $spec|cut -d'|' -f1); BBT=$(echo $spec|cut -d'|' -f2)
  ATTR=$(echo $spec|cut -d'|' -f3); KS=$(echo $spec|cut -d'|' -f4); RN=$(echo $spec|cut -d'|' -f5)
  CK=$(find results/checkpoints_fix -path "*${BBT}*${ATTR}*best_model.pt" | head -1)
  [ -z "$CK" ] && continue
  python -m evaluation.eval_all --dataset fairface --model_path "$CK" --backbone "$BB" \
    --attribute "$ATTR" > "fix_eval_${BBT}_${ATTR}.txt" 2>&1
  BB="$BB" BBT="$BBT" ATTR="$ATTR" KS="$KS" RN="$RN" CK="$CK" python3 - << 'PY' >> fix_iterative.txt 2>&1
import torch, os
from omegaconf import OmegaConf
from models.fairclip import FairCLIP
from data.datasets import FaceDataset, collate_dict
from torch.utils.data import DataLoader
from evaluation.metrics.all_metrics import maxskew_ndkl_zhang, ZHANG_NEUTRAL_QUERIES
bb=os.environ["BB"]; bbt=os.environ["BBT"]; attr=os.environ["ATTR"]
ks=[int(x) for x in os.environ["KS"].split(",")]; rn=int(os.environ["RN"]); ck=os.environ["CK"]
cfg=OmegaConf.load("configs/datasets/local_paths.yaml")
m=FairCLIP(model_name=bb,device="cuda"); m.load(ck); m.eval()
ds=FaceDataset(cfg.fairface.manifest,split="val",train=False)
ld=DataLoader(ds,batch_size=16 if "L_14" in bbt else 8,shuffle=False,collate_fn=collate_dict,num_workers=2)
img,labs=[],[]
with torch.no_grad():
    for b in ld:
        img.append(m.backbone.encode_images(b["image"].to("cuda")).cpu())
        labs.append(b[attr].clone().detach())
img=torch.cat(img).float(); labs=torch.cat(labs); img=img/img.norm(dim=-1,keepdim=True)
txt=m.encode_text(ZHANG_NEUTRAL_QUERIES).cpu().float(); txt=txt/txt.norm(dim=-1,keepdim=True)
def fit(E,l,k):
    g=sorted(l.unique().tolist()); c=torch.stack([E[l==x].mean(0) for x in g])
    _,_,Vt=torch.linalg.svd(c-c.mean(0,keepdim=True),full_matrices=False); return Vt.T[:,:k]
def proj(E,B):
    d=E-(E@B)@B.T; return d/d.norm(dim=-1,keepdim=True).clamp(min=1e-8)
tgt={"gender":(0.106,0.035),"age":(0.515,0.289)}[attr]
print(f"--- {bbt} {attr} (target MS<{tgt[0]} ND<{tgt[1]}) ---", flush=True)
for k in ks:
    ei,et=img.clone(),txt.clone()
    for r in range(1,rn+1):
        B=fit(ei,labs,k); ei=proj(ei,B); et=proj(et,B)
        ms,nd=maxskew_ndkl_zhang(et,ei,labs,k_skew=1000,k_ndkl=1000)
        flag="BOTH" if (ms<tgt[0] and nd<tgt[1]) else ("MS" if ms<tgt[0] else "-")
        print(f"k={k} r={r}: MaxSkew={ms:.4f} NDKL={nd:.4f} {flag}", flush=True)
PY
done
echo done > two_fixes_done.txt
