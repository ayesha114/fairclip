#!/usr/bin/env bash
cd /home/ayeshakalsoom/Downloads/thesis
source /home/ayeshakalsoom/Downloads/fairclip-thesis/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=retrain_retrfix.log
echo "START $(date)" >> $LOG
run () {
  bb=$1; attr=$2; bbs=$(echo $bb | sed 's#/#_#')
  out="results/checkpoints_retrfix/fairface/${bbs}/${attr}/seed42/best_model.pt"
  if [ -f "$out" ]; then echo "SKIP $bb $attr" | tee -a $LOG; return; fi
  # clear stale latest.pt to avoid bad-resume OOM
  rm -f "results/checkpoints_retrfix/fairface/${bbs}/${attr}/seed42/latest.pt"
  lf=0.5; lr=5e-6; ret=2.0
  [ "$attr" = "race" ] && { lf=2.0; lr=1e-5; ret=1.0; }
  echo "TRAIN $bb $attr $(date)" | tee -a $LOG
  python -m training.train --dataset fairface --attribute $attr --backbone $bb \
    --epochs 30 --patience 8 --batch_size 64 --lambda_fair $lf --lambda_retrieval $ret \
    --lr $lr --n_bias_dirs 8 --seed 42 \
    --output_dir results/checkpoints_retrfix > "train_retrfix_${bbs}_${attr}.log" 2>&1
  echo "DONE $bb $attr $(date)" | tee -a $LOG
}
for bb in "ViT-B/16" "ViT-L/14" "ViT-H/14"; do
  for attr in race age; do
    run "$bb" $attr
  done
done
echo "ALL DONE $(date)" | tee -a $LOG
echo done > retrfix_complete.flag
