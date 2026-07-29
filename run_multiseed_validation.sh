#!/usr/bin/env bash
cd /home/ayeshakalsoom/Downloads/thesis
source /home/ayeshakalsoom/Downloads/fairclip-thesis/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG=multiseed_validation_progress.log
echo "START $(date)" > $LOG

# GUARD: refuse to start if another training process already exists
OTHER=$(pgrep -f "training.train" | grep -v $$ | wc -l)
if [ "$OTHER" -gt 0 ]; then
  echo "ABORT: another training.train process is running. Kill it first." | tee -a $LOG
  exit 1
fi

declare -A LAM=( ["gender"]="0.01" ["age"]="0.5" ["race"]="2.0" )
declare -A NBD=( ["gender"]="5" ["age"]="5" ["race"]="5" )

wait_for_gpu () {
  # wait until at least 5000 MiB free (or 20 tries)
  for i in $(seq 1 20); do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$FREE" -ge 5000 ]; then return 0; fi
    echo "  waiting for GPU (free=${FREE}MiB)..." | tee -a $LOG
    sleep 30
  done
  return 0
}

train_one () {
  local bb="$1" bbt="$2" attr="$3" seed="$4" bs="$5"
  local lam="${LAM[$attr]}" nbd="${NBD[$attr]}"
  if [ "$bbt" = "ViT-L_14" ] && [ "$attr" = "gender" ]; then lam="0.003"; nbd="10"; fi
  local ck="results/checkpoints/fairface/${bbt}/${attr}/seed${seed}/best_model.pt"
  if [ -f "$ck" ]; then echo "SKIP (exists): $bbt $attr seed$seed" | tee -a $LOG; return; fi

  wait_for_gpu
  echo "TRAIN: $bbt $attr seed$seed lam=$lam nbd=$nbd bs=$bs $(date)" | tee -a $LOG
  python -m training.train --dataset fairface --attribute "$attr" --backbone "$bb" \
    --epochs 30 --patience 3 --batch_size "$bs" --lambda_fair "$lam" \
    --n_bias_dirs "$nbd" --seed "$seed" > "train_${bbt}_${attr}_seed${seed}.log" 2>&1

  # OOM fallback: retry at batch 8
  if [ ! -f "$ck" ] && grep -q "OutOfMemoryError" "train_${bbt}_${attr}_seed${seed}.log"; then
    echo "RETRY at batch 8: $bbt $attr seed$seed" | tee -a $LOG
    wait_for_gpu
    python -m training.train --dataset fairface --attribute "$attr" --backbone "$bb" \
      --epochs 30 --patience 3 --batch_size 8 --lambda_fair "$lam" \
      --n_bias_dirs "$nbd" --seed "$seed" > "train_${bbt}_${attr}_seed${seed}.log" 2>&1
  fi

  if [ -f "$ck" ]; then
    echo "DONE train: $bbt $attr seed$seed" | tee -a $LOG
    python -m evaluation.eval_all --dataset fairface --model_path "$ck" \
      --backbone "$bb" --attribute "$attr" > "eval_${bbt}_${attr}_seed${seed}.log" 2>&1
    echo "DONE eval:  $bbt $attr seed$seed" | tee -a $LOG
  else
    echo "FAIL: $bbt $attr seed$seed (see train log)" | tee -a $LOG
  fi
}

for seed in 123 456; do
  for attr in gender age race; do
    train_one "ViT-L/14" "ViT-L_14" "$attr" "$seed" 16
  done
done

echo "ALL DONE $(date)" | tee -a $LOG
echo done > multiseed_validation_complete.flag
