#!/usr/bin/env bash
cd /home/ayeshakalsoom/Downloads/thesis
source /home/ayeshakalsoom/Downloads/fairclip-thesis/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=ablations_clean.log
echo "START $(date)" > $LOG
run_abl () {
  NAME=$1; shift
  ck="results/checkpoints_abl_${NAME}/fairface/ViT-B_32/race/seed42/best_model.pt"
  if [ -f "$ck" ]; then echo "SKIP exists: $NAME" | tee -a $LOG; return; fi
  echo "TRAIN ABL: $NAME $(date)" | tee -a $LOG
  python -m training.train --dataset fairface --attribute race --backbone ViT-B/32 \
    --epochs 30 --patience 8 --batch_size 64 --lambda_fair 2.0 --n_bias_dirs 5 --seed 42 \
    --output_dir "results/checkpoints_abl_${NAME}" "$@" > "abl_${NAME}.txt" 2>&1
  echo "DONE: $NAME $(date)" | tee -a $LOG
}
run_abl no_fairloss   --lambda_fair 0.0
run_abl no_temp       --no_adaptive_temp
run_abl no_trainproj  --no_train_proj
run_abl no_procrustes --no_procrustes
run_abl no_subspace   --no_subspace
echo "ALL DONE $(date)" | tee -a $LOG
echo done > ablations_clean_complete.flag
