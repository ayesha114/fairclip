#!/usr/bin/env bash
# Waits for backbones to finish, then runs utility evals (ImageNetV2 + Flickr)
cd /home/ayeshakalsoom/Downloads/thesis
source /home/ayeshakalsoom/Downloads/fairclip-thesis/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while [ ! -f backbones_done.txt ]; do sleep 300; done
echo "backbones done — starting utility evals"
for bb in "ViT-B/32" "ViT-B/16" "ViT-L/14"; do
  bbt=$(echo "$bb" | tr '/' '_')
  # Baseline utility (once per backbone)
  python eval_imagenetv2.py --baseline --backbone "$bb" > "util_inv2_${bbt}_baseline.txt" 2>&1
  # FairCLIP utility per attribute
  for attr in gender age race; do
    CKPT="results/checkpoints/fairface/${bbt}/${attr}/seed42/best_model.pt"
    [ -f "$CKPT" ] && python eval_imagenetv2.py --model_path "$CKPT" --backbone "$bb" \
      > "util_inv2_${bbt}_${attr}.txt" 2>&1
  done
done
echo done > utility_done.txt
# Flickr TR/IR (appended stage)
for bb in "ViT-B/32" "ViT-B/16" "ViT-L/14"; do
  bbt=$(echo "$bb" | tr '/' '_')
  python eval_flickr.py --baseline --backbone "$bb" --tag "baseline_${bbt}" >> flickr_log.txt 2>&1
  for attr in gender age race; do
    CKPT="results/checkpoints/fairface/${bbt}/${attr}/seed42/best_model.pt"
    [ -f "$CKPT" ] && python eval_flickr.py --model_path "$CKPT" --backbone "$bb" --tag "fairclip_${bbt}_${attr}" >> flickr_log.txt 2>&1
  done
done
echo done > flickr_done.txt
