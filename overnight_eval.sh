#!/usr/bin/env bash
cd /home/ayeshakalsoom/Downloads/thesis
source /home/ayeshakalsoom/Downloads/fairclip-thesis/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=overnight_results.txt
touch $OUT
evalcell () {
  bb=$1; attr=$2; seed=$3; bbs=$(echo $bb | sed 's#/#_#')
  ck="results/checkpoints_procfix/fairface/${bbs}/${attr}/seed${seed}/best_model.pt"
  [ -f "$ck" ] || return
  grep -q "^\[$bb $attr s$seed\]" $OUT && { echo "SKIP $bb $attr s$seed"; return; }
  f=$(python eval_methodology_correct.py --attribute $attr --backbone "$bb" --seed $seed --n_dirs 8 --model_path "$ck" 2>/dev/null | grep -oE "DPG=[0-9.]+ EOD=[0-9.]+ RBS=[0-9.e-]+")
  ms=$(python eval_retrieval_clean.py --attribute $attr --backbone "$bb" --seed $seed --rounds 1 --n_dirs 8 --model_path "$ck" 2>/dev/null | grep -oE "MaxSkew=[0-9.]+ NDKL=[0-9.]+")
  echo "[$bb $attr s$seed] $f | $ms" >> $OUT
  echo "DONE $bb $attr s$seed"
}
for seed in 42 123 456; do
  for bb in "ViT-B/32" "ViT-B/16" "ViT-L/14" "ViT-H/14"; do
    for attr in gender age race; do
      evalcell "$bb" $attr $seed
    done
  done
done
echo "ALL DONE" >> $OUT
