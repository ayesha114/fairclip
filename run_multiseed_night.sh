#!/usr/bin/env bash
cd /home/ayeshakalsoom/Downloads/thesis
source /home/ayeshakalsoom/Downloads/fairclip-thesis/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG=multiseed_night_log.txt
echo "START $(date)" > $LOG

# recipe: attr -> "lambda n_bias_dirs"
declare -A LAM=( ["gender"]="0.01" ["age"]="0.5" ["race"]="2.0" )
declare -A NBD=( ["gender"]="5" ["age"]="5" ["race"]="5" )

train_one () {
  local bb="$1" bbt="$2" attr="$3" seed="$4" bs="$5"
  local lam="${LAM[$attr]}" nbd="${NBD[$attr]}"
  # L/14 gender uses the fixed recipe
  if [ "$bbt" = "ViT-L_14" ] && [ "$attr" = "gender" ]; then
    lam="0.003"; nbd="10"
  fi
  local ckdir="results/checkpoints/fairface/${bbt}/${attr}/seed${seed}"
  local ck="${ckdir}/best_model.pt"
  if [ -f "$ck" ]; then
    echo "SKIP (exists): $bbt $attr seed$seed" | tee -a $LOG
    return
  fi
  echo "TRAIN: $bbt $attr seed$seed lam=$lam nbd=$nbd bs=$bs $(date)" | tee -a $LOG
  python -m training.train --dataset fairface --attribute "$attr" --backbone "$bb" \
    --epochs 30 --patience 3 --batch_size "$bs" --lambda_fair "$lam" \
    --n_bias_dirs "$nbd" --seed "$seed" > "ms_train_${bbt}_${attr}_s${seed}.txt" 2>&1
  if [ -f "$ck" ]; then
    echo "DONE train: $bbt $attr seed$seed" | tee -a $LOG
    python -m evaluation.eval_all --dataset fairface --model_path "$ck" \
      --backbone "$bb" --attribute "$attr" > "ms_eval_${bbt}_${attr}_s${seed}.txt" 2>&1
    echo "DONE eval:  $bbt $attr seed$seed" | tee -a $LOG
  else
    echo "FAIL (no checkpoint): $bbt $attr seed$seed" | tee -a $LOG
  fi
}

# ViT-B/16 (batch 32) seeds 123,456
for seed in 123 456; do
  for attr in gender age race; do
    train_one "ViT-B/16" "ViT-B_16" "$attr" "$seed" 32
  done
done

# ViT-L/14 (batch 16) seeds 123,456
for seed in 123 456; do
  for attr in gender age race; do
    train_one "ViT-L/14" "ViT-L_14" "$attr" "$seed" 16
  done
done

echo "ALL DONE $(date)" | tee -a $LOG
echo done > multiseed_night_done.txt
