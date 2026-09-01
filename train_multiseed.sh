#!/usr/bin/env bash
cd /home/ayeshakalsoom/Downloads/thesis
source /home/ayeshakalsoom/Downloads/fairclip-thesis/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=multiseed.log
echo "START $(date)" >> $LOG
run () {
  bb=$1; attr=$2; seed=$3; bbs=$(echo $bb | sed 's#/#_#')
  out="results/checkpoints_procfix/fairface/${bbs}/${attr}/seed${seed}/best_model.pt"
  [ -f "$out" ] && { echo "SKIP $bb $attr s$seed" | tee -a $LOG; return; }
  rm -f "results/checkpoints_procfix/fairface/${bbs}/${attr}/seed${seed}/latest.pt"
  lf=2.0; [ "$attr" = "gender" ] && lf=0.01; [ "$attr" = "age" ] && lf=0.5
  echo "TRAIN $bb $attr s$seed $(date)" | tee -a $LOG
  python -m training.train --dataset fairface --attribute $attr --backbone $bb \
    --epochs 30 --patience 8 --batch_size 64 --lambda_fair $lf --n_bias_dirs 8 --seed $seed \
    --output_dir results/checkpoints_procfix > "train_ms_${bbs}_${attr}_s${seed}.log" 2>&1
  echo "DONE $bb $attr s$seed $(date)" | tee -a $LOG
}
# smaller/faster backbones first so you get results early
for seed in 123 456; do
  for bb in "ViT-B/32" "ViT-B/16" "ViT-L/14"; do
    for attr in gender age race; do
      run "$bb" $attr $seed
    done
  done
done
echo "ALL DONE $(date)" | tee -a $LOG
echo done > multiseed_complete.flag
