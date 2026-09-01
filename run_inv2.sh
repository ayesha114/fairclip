#!/usr/bin/env bash
cd /home/ayeshakalsoom/Downloads/thesis
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=results/IMAGENETV2.txt
touch $OUT
for bb in "ViT-B/32" "ViT-B/16" "ViT-L/14" "ViT-H/14"; do
  bbs=$(echo $bb | sed 's#/#_#')
  grep -q "^\[$bb BASE\]" $OUT || {
    r=$(python eval_imagenetv2.py --backbone "$bb" --baseline 2>&1 | grep -oE "Top-1: [0-9.]+%   Top-5: [0-9.]+%")
    echo "[$bb BASE] $r" >> $OUT; echo "DONE base $bb"; }
  for at in gender age race; do
    ck="results/checkpoints/fairface/${bbs}/${at}/seed42/best_model.pt"
    [ -f "$ck" ] || continue
    grep -q "^\[$bb $at\]" $OUT && continue
    r=$(python eval_imagenetv2.py --backbone "$bb" --model_path "$ck" 2>&1 | grep -oE "Top-1: [0-9.]+%   Top-5: [0-9.]+%")
    echo "[$bb $at] $r" >> $OUT; echo "DONE $bb $at"
  done
done
echo "IMAGENETV2 FINISHED"
