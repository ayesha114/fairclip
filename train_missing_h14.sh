#!/usr/bin/env bash
cd /home/ayeshakalsoom/Downloads/thesis
source /home/ayeshakalsoom/Downloads/fairclip-thesis/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=train_missing_h14.log
echo "START $(date)" > $LOG
declare -A LAM=( [gender]=0.01 [age]=0.5 [race]=1.0 )
for combo in "gender 456" "age 123" "age 456" "race 123" "race 456"; do
  set -- $combo; attr=$1; seed=$2
  ck="results/checkpoints/fairface/ViT-H_14/${attr}/seed${seed}/best_model.pt"
  if [ -f "$ck" ]; then echo "SKIP exists: $attr seed$seed" | tee -a $LOG; continue; fi
  echo "TRAIN: H/14 $attr seed$seed $(date)" | tee -a $LOG
  python -m training.train --dataset fairface --attribute $attr --backbone ViT-H/14 \
    --epochs 30 --patience 5 --batch_size 8 --lambda_fair ${LAM[$attr]} \
    --n_bias_dirs 5 --seed $seed > "train_h14_${attr}_seed${seed}.log" 2>&1
  echo "DONE: H/14 $attr seed$seed $(date)" | tee -a $LOG
done
echo "ALL DONE $(date)" | tee -a $LOG
echo done > train_missing_h14_complete.flag
