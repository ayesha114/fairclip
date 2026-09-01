#!/usr/bin/env bash
cd /home/ayeshakalsoom/Downloads/thesis
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=results/FF_MULTISEED_MS.txt
touch $OUT
declare -A K=( [gender]=2 [age]=8 [race]=6 )
declare -A R=( [gender]=2 [age]=3 [race]=5 )
for sd in 123 456; do
for bb in "ViT-B/32" "ViT-B/16" "ViT-L/14" "ViT-H/14"; do
for at in gender age race; do
  bbs=$(echo $bb | sed 's#/#_#')
  grep -q "^\[$bb $at s$sd\]" $OUT && continue
  ck="results/checkpoints/fairface/${bbs}/${at}/seed${sd}/best_model.pt"
  [ -f "$ck" ] || { echo "MISS $bb $at s$sd"; continue; }
  r=$(python eval_retrieval_clean.py --attribute $at --backbone "$bb" --seed $sd \
      --rounds ${R[$at]} --n_dirs ${K[$at]} --model_path "$ck" | grep "FairCLIP  " | grep -oE "MaxSkew=[0-9.]+ NDKL=[0-9.]+")
  echo "[$bb $at s$sd] $r" >> $OUT
  echo "DONE $bb $at s$sd $r"
done; done; done
echo FINISHED
