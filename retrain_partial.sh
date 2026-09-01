#!/usr/bin/env bash
cd /home/ayeshakalsoom/Downloads/thesis
source /home/ayeshakalsoom/Downloads/fairclip-thesis/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=retrain_partial.log
echo "START $(date)" >> $LOG
run () {
  bb=$1; attr=$2; ret=$3; bbs=$(echo $bb | sed 's#/#_#')
  out="results/checkpoints_partial/fairface/${bbs}/${attr}/seed42/best_model.pt"
  [ -f "$out" ] && { echo "SKIP $bb $attr" | tee -a $LOG; return; }
  rm -f "results/checkpoints_partial/fairface/${bbs}/${attr}/seed42/latest.pt"
  lf=0.5; lr=5e-6; [ "$attr" = "race" ] && { lf=2.0; lr=1e-5; }
  echo "TRAIN $bb $attr ret=$ret $(date)" | tee -a $LOG
  python -m training.train --dataset fairface --attribute $attr --backbone $bb \
    --epochs 30 --patience 8 --batch_size 64 --lambda_fair $lf --lambda_retrieval $ret \
    --lr $lr --n_bias_dirs 8 --seed 42 \
    --output_dir results/checkpoints_partial > "train_partial_${bbs}_${attr}.log" 2>&1
  echo "DONE $bb $attr $(date)" | tee -a $LOG
}
# Higher lambda_retrieval for the partial cells
run "ViT-B/16" age 5.0
run "ViT-L/14" race 3.0
run "ViT-H/14" race 3.0
run "ViT-H/14" age 5.0
echo "ALL DONE $(date)" | tee -a $LOG
echo done > partial_complete.flag
