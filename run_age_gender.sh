#!/usr/bin/env bash
cd /home/ayeshakalsoom/Downloads/thesis
source /home/ayeshakalsoom/Downloads/fairclip-thesis/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# AGE: locked recipe (50/50 + lambda 0.5 + train-time projection)
rm -rf results/checkpoints/fairface/ViT-B_32/age/seed42
python -m training.train --dataset fairface --attribute age --backbone ViT-B/32 \
  --epochs 30 --patience 8 --batch_size 64 --lambda_fair 0.5 --n_bias_dirs 5 --seed 42 > age_final.txt 2>&1
# GENDER: lambda 0.01 (baseline already fair, minimal intervention)
rm -rf results/checkpoints/fairface/ViT-B_32/gender/seed42
python -m training.train --dataset fairface --attribute gender --backbone ViT-B/32 \
  --epochs 30 --patience 8 --batch_size 64 --lambda_fair 0.01 --n_bias_dirs 5 --seed 42 > gender_final.txt 2>&1
echo "BOTH DONE" > age_gender_done.txt
