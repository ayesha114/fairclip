#!/usr/bin/env bash
cd /home/ayeshakalsoom/Downloads/thesis
source /home/ayeshakalsoom/Downloads/fairclip-thesis/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
declare -A LAMBDA=( ["gender"]=0.01 ["age"]=0.5 ["race"]=2.0 )
for seed in 123 456; do
  for attr in gender age race; do
    CKPT="results/checkpoints/fairface/ViT-B_32/${attr}/seed${seed}"
    [ -f "$CKPT/best_model.pt" ] && { echo "skip $attr seed$seed"; continue; }
    rm -rf "$CKPT"
    python -m training.train --dataset fairface --attribute "$attr" --backbone ViT-B/32 \
      --epochs 30 --patience 8 --batch_size 64 --lambda_fair "${LAMBDA[$attr]}" \
      --n_bias_dirs 5 --seed "$seed" > "ms_${attr}_s${seed}.txt" 2>&1
    python -m evaluation.eval_all --dataset fairface \
      --model_path "$CKPT/best_model.pt" --backbone ViT-B/32 --attribute "$attr" \
      > "ms_eval_${attr}_s${seed}.txt" 2>&1
  done
done
echo done > multiseed_done.txt
