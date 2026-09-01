#!/usr/bin/env bash
cd /home/ayeshakalsoom/Downloads/thesis
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=results/ABLATION_MS_FIXED.txt
touch $OUT
declare -A K=( [gender]=2 [age]=8 [race]=6 )
declare -A R=( [gender]=2 [age]=3 [race]=5 )
for cfg in FULL no_fairloss no_temp no_trainproj no_procrustes no_subspace; do
for at in gender age race; do
  grep -q "^\[$cfg $at\]" $OUT && continue
  if [ "$cfg" = "FULL" ]; then
    ck="results/checkpoints_procfix/fairface/ViT-B_32/${at}/seed42/best_model.pt"
  else
    ck="results/checkpoints_ablfix_${cfg}/fairface/ViT-B_32/${at}/seed42/best_model.pt"
  fi
  [ -f "$ck" ] || continue
  r=$(python eval_retrieval_clean.py --attribute $at --backbone ViT-B/32 --seed 42 \
      --rounds ${R[$at]} --n_dirs ${K[$at]} --model_path "$ck" | grep "FairCLIP  " | grep -oE "MaxSkew=[0-9.]+ NDKL=[0-9.]+")
  echo "[$cfg $at] $r" >> $OUT
  echo "DONE $cfg $at $r"
done; done
echo FINISHED
