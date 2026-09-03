#!/usr/bin/env bash
cd /home/ayeshakalsoom/Downloads/thesis
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f queue_utility_seeds.sh > /dev/null; do sleep 60; done
OUT=results/BASELINE_DPG_MISSING.txt; touch $OUT
for spec in "ViT-B/16 gender" "ViT-H/14 gender" "ViT-H/14 age" "ViT-H/14 race"; do
  set -- $spec; bb=$1; at=$2
  grep -q "^\[$bb $at\]" $OUT && continue
  r=$(python eval_methodology_correct.py --attribute $at --backbone "$bb" --seed 42 --n_dirs 8 --baseline | grep -oE "DPG=[0-9.]+ EOD=[0-9.]+ RBS=[0-9.e-]+")
  echo "[$bb $at] $r" >> $OUT; echo "DONE base $bb $at $r"
done
echo BASELINE_DPG_FINISHED
