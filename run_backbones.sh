#!/usr/bin/env bash
cd /home/ayeshakalsoom/Downloads/thesis
source /home/ayeshakalsoom/Downloads/fairclip-thesis/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
declare -A LAMBDA=( ["gender"]=0.01 ["age"]=0.5 ["race"]=2.0 )
for bb in "ViT-B/16" "ViT-L/14"; do
  bbt=$(echo "$bb" | tr '/' '_')
  for attr in gender age race; do
    CKPT="results/checkpoints/fairface/${bbt}/${attr}/seed42"
    [ -f "$CKPT/best_model.pt" ] && { echo "skip $bb $attr"; continue; }
    rm -rf "$CKPT"
    BS=32; [ "$bb" = "ViT-L/14" ] && BS=16
    python -m training.train --dataset fairface --attribute "$attr" --backbone "$bb" \
      --epochs 30 --patience 8 --batch_size $BS --lambda_fair "${LAMBDA[$attr]}" \
      --n_bias_dirs 5 --seed 42 > "bb_${bbt}_${attr}.txt" 2>&1
    python -m evaluation.eval_all --dataset fairface \
      --model_path "$CKPT/best_model.pt" --backbone "$bb" --attribute "$attr" \
      > "bb_eval_${bbt}_${attr}.txt" 2>&1
  done
done
echo done > backbones_done.txt
