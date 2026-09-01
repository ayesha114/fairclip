#!/usr/bin/env bash
cd /home/ayeshakalsoom/Downloads/thesis
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=results/UTK_dpg_eod_rbs.txt
touch $OUT
for bb in "ViT-B/32" "ViT-B/16" "ViT-L/14" "ViT-H/14"; do
for at in gender age race; do
  bbs=$(echo $bb | sed 's#/#_#')
  grep -q "^\[$bb $at BASE\]" $OUT || {
    r=$(python cross_dpg.py --attribute $at --backbone "$bb" --baseline | grep -oE "DPG=[0-9.]+ EOD=[0-9.]+ RBS=[0-9.e-]+")
    echo "[$bb $at BASE] $r" >> $OUT; echo "DONE base $bb $at"; }
  for sd in 42 123 456; do
    ck="results/checkpoints/fairface/${bbs}/${at}/seed${sd}/best_model.pt"
    [ -f "$ck" ] || continue
    grep -q "^\[$bb $at s$sd\]" $OUT && continue
    r=$(python cross_dpg.py --attribute $at --backbone "$bb" --seed $sd --n_dirs 8 --train_cap 5000 --model_path "$ck" | grep -oE "DPG=[0-9.]+ EOD=[0-9.]+ RBS=[0-9.e-]+")
    echo "[$bb $at s$sd] $r" >> $OUT; echo "DONE $bb $at s$sd"
  done
done; done
echo "UTK DPG FINISHED"
