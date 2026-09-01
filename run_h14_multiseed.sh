#!/usr/bin/env bash
cd /home/ayeshakalsoom/Downloads/thesis
source /home/ayeshakalsoom/Downloads/fairclip-thesis/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=h14_ms.log
echo "START $(date)" >> $LOG
for seed in 123 456; do
  for attr in gender age race; do
    out="results/checkpoints_procfix/fairface/ViT-H_14/${attr}/seed${seed}/best_model.pt"
    if [ -f "$out" ]; then echo "SKIP H/14 $attr s$seed" | tee -a $LOG; continue; fi
    # clean GPU before each H/14 cell (prevents zombie OOM)
    pkill -9 -f "training.train"; sleep 8
    rm -f "results/checkpoints_procfix/fairface/ViT-H_14/${attr}/seed${seed}/latest.pt"
    lf=2.0; [ "$attr" = "gender" ] && lf=0.01; [ "$attr" = "age" ] && lf=0.5
    echo "TRAIN H/14 $attr s$seed $(date)" | tee -a $LOG
    python -m training.train --dataset fairface --attribute $attr --backbone ViT-H/14 \
      --epochs 30 --patience 3 --batch_size 64 --lambda_fair $lf --n_bias_dirs 8 --seed $seed \
      --output_dir results/checkpoints_procfix > "train_ms_ViT-H_14_${attr}_s${seed}.log" 2>&1
    echo "DONE H/14 $attr s$seed $(date)" | tee -a $LOG
  done
done
echo "H14 ALL DONE $(date)" | tee -a $LOG
echo done > h14_ms_complete.flag
