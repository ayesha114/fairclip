#!/usr/bin/env bash
cd /home/ayeshakalsoom/Downloads/thesis
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f rbs_multiseed.sh > /dev/null; do sleep 60; done

# --- Flickr retrieval, seeds 123 & 456 ---
OUT=results/RETRIEVAL_SEEDS.txt; touch $OUT
for sd in 123 456; do
for bb in "ViT-B/32" "ViT-B/16" "ViT-L/14" "ViT-H/14"; do
for at in gender age race; do
  bbs=$(echo $bb | sed 's#/#_#')
  grep -q "^\[$bb $at s$sd\]" $OUT && continue
  ck="results/checkpoints_procfix/fairface/${bbs}/${at}/seed${sd}/best_model.pt"
  [ -f "$ck" ] || { echo "MISS $bb $at s$sd"; continue; }
  r=$(python eval_frozen_final.py --backbone "$bb" --model_path "$ck" | grep -oE "TR@1=.*")
  echo "[$bb $at s$sd] $r" >> $OUT; echo "DONE retr $bb $at s$sd"
done; done; done

# --- ImageNetV2, seeds 123 & 456 ---
OUT2=results/IMAGENETV2_SEEDS.txt; touch $OUT2
for sd in 123 456; do
for bb in "ViT-B/32" "ViT-B/16" "ViT-L/14" "ViT-H/14"; do
for at in gender age race; do
  bbs=$(echo $bb | sed 's#/#_#')
  grep -q "^\[$bb $at s$sd\]" $OUT2 && continue
  ck="results/checkpoints/fairface/${bbs}/${at}/seed${sd}/best_model.pt"
  [ -f "$ck" ] || { echo "MISS in2 $bb $at s$sd"; continue; }
  r=$(python eval_imagenetv2.py --backbone "$bb" --model_path "$ck" | grep -oE "Top-1: [0-9.]+%   Top-5: [0-9.]+%")
  echo "[$bb $at s$sd] $r" >> $OUT2; echo "DONE in2 $bb $at s$sd"
done; done; done
echo UTILITY_SEEDS_FINISHED
