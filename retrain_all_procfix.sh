#!/usr/bin/env bash
cd /home/ayeshakalsoom/Downloads/thesis
source /home/ayeshakalsoom/Downloads/fairclip-thesis/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=retrain_all_procfix.log
echo "START $(date)" > $LOG
run () {
  bb=$1; attr=$2; seed=$3
  bbs=$(echo $bb | sed 's#/#_#')
  out="results/checkpoints_procfix/fairface/${bbs}/${attr}/seed${seed}"
  if [ -f "$out/best_model.pt" ]; then echo "SKIP $bb $attr $seed" | tee -a $LOG; return; fi
  echo "TRAIN $bb $attr seed$seed $(date)" | tee -a $LOG
  lf=2.0; [ "$attr" = "gender" ] && lf=0.01
  python -m training.train --dataset fairface --attribute $attr --backbone $bb \
    --epochs 30 --patience 8 --batch_size 64 --lambda_fair $lf --n_bias_dirs 8 --seed $seed \
    --output_dir "results/checkpoints_procfix" > "train_procfix_${bbs}_${attr}_${seed}.log" 2>&1
  echo "DONE $bb $attr seed$seed $(date)" | tee -a $LOG
}
for bb in "ViT-B/32" "ViT-B/16" "ViT-L/14" "ViT-H/14"; do
  for attr in gender age race; do
    run "$bb" $attr 42
  done
done
echo "ALL DONE $(date)" | tee -a $LOG
echo done > retrain_procfix_complete.flag
