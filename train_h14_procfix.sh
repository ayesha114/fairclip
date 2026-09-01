#!/usr/bin/env bash
cd /home/ayeshakalsoom/Downloads/thesis
source /home/ayeshakalsoom/Downloads/fairclip-thesis/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for attr in gender age race; do
  lf=2.0; [ "$attr" = "gender" ] && lf=0.01
  out="results/checkpoints_procfix/fairface/ViT-H_14/${attr}/seed42/best_model.pt"
  [ -f "$out" ] && { echo "SKIP H/14 $attr"; continue; }
  echo "TRAIN H/14 $attr $(date)"
  python -m training.train --dataset fairface --attribute $attr --backbone ViT-H/14 \
    --epochs 30 --patience 8 --batch_size 64 --lambda_fair $lf --n_bias_dirs 8 --seed 42 \
    --output_dir results/checkpoints_procfix > "train_procfix_ViT-H_14_${attr}_42.log" 2>&1
  echo "DONE H/14 $attr $(date)"
done
echo "H14 DONE" > h14_procfix_done.flag
