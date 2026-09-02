#!/usr/bin/env bash
cd /home/ayeshakalsoom/Downloads/thesis
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=results/RETRIEVAL_ALL_K.txt; touch $OUT
for bb in "ViT-B/32" "ViT-B/16" "ViT-L/14" "ViT-H/14"; do
  bbs=$(echo $bb | sed 's#/#_#')
  grep -q "^\[$bb BASE\]" $OUT || {
    r=$(python eval_flickr.py --backbone "$bb" --baseline --tag base | grep -oE "TR@5=[0-9.]+ IR@5=[0-9.]+ TR@1=[0-9.]+ IR@1=[0-9.]+")
    echo "[$bb BASE] $r" >> $OUT; echo "DONE base $bb"; }
  for at in gender age race; do
    grep -q "^\[$bb $at\]" $OUT && continue
    r=$(python eval_frozen_final.py --backbone "$bb" --model_path results/checkpoints_procfix/fairface/${bbs}/${at}/seed42/best_model.pt | grep -oE "TR@1=.*")
    echo "[$bb $at] $r" >> $OUT; echo "DONE $bb $at"
  done
done
echo FINISHED
