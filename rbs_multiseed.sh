#!/usr/bin/env bash
cd /home/ayeshakalsoom/Downloads/thesis
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f trall.sh > /dev/null; do sleep 60; done
OUT=results/FF_MULTISEED_DPG_EOD_RBS.txt; touch $OUT
for sd in 42 123 456; do
for bb in "ViT-B/32" "ViT-B/16" "ViT-L/14" "ViT-H/14"; do
for at in gender age race; do
  bbs=$(echo $bb | sed 's#/#_#')
  grep -q "^\[$bb $at s$sd\]" $OUT && continue
  ck="results/checkpoints_procfix/fairface/${bbs}/${at}/seed${sd}/best_model.pt"
  [ -f "$ck" ] || { echo "MISS $bb $at s$sd"; continue; }
  r=$(python eval_methodology_correct.py --attribute $at --backbone "$bb" --seed $sd --n_dirs 8 --model_path "$ck" | grep -oE "DPG=[0-9.]+ EOD=[0-9.]+ RBS=[0-9.e-]+")
  echo "[$bb $at s$sd] $r" >> $OUT
  echo "DONE $bb $at s$sd $r"
done; done; done
echo RBS_MULTISEED_FINISHED
