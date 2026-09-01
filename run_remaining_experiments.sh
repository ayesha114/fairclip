#!/usr/bin/env bash
cd /home/ayeshakalsoom/Downloads/thesis
source /home/ayeshakalsoom/Downloads/fairclip-thesis/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=remaining_experiments_progress.log
echo "START $(date)" > $LOG

declare -A LAM=( ["gender"]="0.01" ["age"]="0.5" ["race"]="2.0" )
declare -A NBD=( ["gender"]="5" ["age"]="5" ["race"]="5" )

wait_for_gpu () {
  for i in $(seq 1 40); do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$FREE" -ge 5000 ] && return 0
    echo "  waiting for GPU (free=${FREE}MiB)..." | tee -a $LOG; sleep 60
  done
}

train_seed () {  # backbone bbt attr seed bs
  local bb="$1" bbt="$2" attr="$3" seed="$4" bs="$5"
  local lam="${LAM[$attr]}" nbd="${NBD[$attr]}"
  [ "$bbt" = "ViT-L_14" ] && [ "$attr" = "gender" ] && { lam="0.003"; nbd="10"; }
  local ck="results/checkpoints/fairface/${bbt}/${attr}/seed${seed}/best_model.pt"
  if [ -f "$ck" ]; then echo "SKIP: $bbt $attr seed$seed" | tee -a $LOG; return; fi
  wait_for_gpu
  echo "TRAIN: $bbt $attr seed$seed $(date)" | tee -a $LOG
  python -m training.train --dataset fairface --attribute "$attr" --backbone "$bb" \
    --epochs 30 --patience 3 --batch_size "$bs" --lambda_fair "$lam" --n_bias_dirs "$nbd" \
    --seed "$seed" > "train_${bbt}_${attr}_seed${seed}.log" 2>&1
  if [ ! -f "$ck" ] && grep -q "OutOfMemory" "train_${bbt}_${attr}_seed${seed}.log"; then
    echo "RETRY bs8: $bbt $attr seed$seed" | tee -a $LOG; wait_for_gpu
    python -m training.train --dataset fairface --attribute "$attr" --backbone "$bb" \
      --epochs 30 --patience 3 --batch_size 8 --lambda_fair "$lam" --n_bias_dirs "$nbd" \
      --seed "$seed" > "train_${bbt}_${attr}_seed${seed}.log" 2>&1
  fi
  if [ -f "$ck" ]; then
    python -m evaluation.eval_all --dataset fairface --model_path "$ck" --backbone "$bb" \
      --attribute "$attr" > "eval_${bbt}_${attr}_seed${seed}.log" 2>&1
    echo "DONE: $bbt $attr seed$seed" | tee -a $LOG
  else echo "FAIL: $bbt $attr seed$seed" | tee -a $LOG; fi
}

ablate () {  # attr ablation_flag name
  local attr="$1" flag="$2" name="$3" lam="${LAM[$attr]}"
  [ "$name" = "no_fairloss" ] && lam="0.0"
  local out="results/checkpoints_abl_${attr}_${name}"
  local ck=$(find "$out" -name best_model.pt 2>/dev/null | head -1)
  if [ -n "$ck" ]; then echo "SKIP abl: $attr $name" | tee -a $LOG; return; fi
  wait_for_gpu
  echo "ABLATE: $attr $name $(date)" | tee -a $LOG
  python -m training.train --dataset fairface --attribute "$attr" --backbone ViT-B/32 \
    --epochs 30 --patience 3 --batch_size 32 --lambda_fair "$lam" --n_bias_dirs 5 --seed 42 \
    $flag --output_dir "$out" > "abl_${attr}_${name}.log" 2>&1
  ck=$(find "$out" -name best_model.pt 2>/dev/null | head -1)
  [ -n "$ck" ] && python -m evaluation.eval_all --dataset fairface --model_path "$ck" \
    --backbone ViT-B/32 --attribute "$attr" > "abl_eval_${attr}_${name}.log" 2>&1
  echo "DONE abl: $attr $name" | tee -a $LOG
}

# ---- 1. H/14 multi-seed (seeds 123, 456) ----
for seed in 123 456; do
  for attr in gender age race; do
    train_seed "ViT-H/14" "ViT-H_14" "$attr" "$seed" 8
  done
done

# ---- 2. Ablations on age and gender (B/32) ----
for attr in age gender; do
  ablate "$attr" "--no_adaptive_temp" "no_temp"
  ablate "$attr" "--no_procrustes"    "no_procrustes"
  ablate "$attr" "--no_train_proj"    "no_trainproj"
  ablate "$attr" "--lambda_fair 0.0"  "no_fairloss"
done

echo "ALL DONE $(date)" | tee -a $LOG
echo done > remaining_experiments_complete.flag
