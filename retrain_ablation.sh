#!/usr/bin/env bash
cd /home/ayeshakalsoom/Downloads/thesis
source /home/ayeshakalsoom/Downloads/fairclip-thesis/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=ablation_procfix.log
echo "START $(date)" >> $LOG
run () {
  name=$1; flag=$2; attr=$3
  lf=2.0; [ "$attr" = "gender" ] && lf=0.01; [ "$attr" = "age" ] && lf=0.5
  [ "$name" = "no_fairloss" ] && lf=0
  out="results/checkpoints_ablfix_${name}/fairface/ViT-B_32/${attr}/seed42/best_model.pt"
  [ -f "$out" ] && { echo "SKIP $name $attr" | tee -a $LOG; return; }
  echo "TRAIN $name $attr $(date)" | tee -a $LOG
  python -m training.train --dataset fairface --attribute $attr --backbone ViT-B/32 \
    --epochs 30 --patience 5 --batch_size 64 --lambda_fair $lf --n_bias_dirs 8 --seed 42 \
    $flag --output_dir results/checkpoints_ablfix_${name} > "train_ablfix_${name}_${attr}.log" 2>&1
  echo "DONE $name $attr $(date)" | tee -a $LOG
}
for attr in gender age race; do
  run no_fairloss   ""                   $attr
  run no_temp       "--no_adaptive_temp" $attr
  run no_trainproj  "--no_train_proj"    $attr
  run no_procrustes "--no_procrustes"    $attr
  run no_subspace   "--no_subspace"      $attr
done
echo "ABL DONE $(date)" | tee -a $LOG
echo done > ablation_complete.flag
