#!/usr/bin/env bash
cd /home/ayeshakalsoom/Downloads/thesis
source /home/ayeshakalsoom/Downloads/fairclip-thesis/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
declare -A LAMBDA=( ["gender"]=0.01 ["age"]=0.5 ["race"]=2.0 )
for attr in gender age race; do
  rm -rf "results/checkpoints/fairface/ViT-H_14/${attr}/seed42"
  python -m training.train --dataset fairface --attribute "$attr" --backbone ViT-H/14 \
    --epochs 30 --patience 3 --batch_size 8 --lambda_fair "${LAMBDA[$attr]}" \
    --n_bias_dirs 5 --seed 42 > "h14_${attr}.txt" 2>&1
done
echo done > h14_done.txt
