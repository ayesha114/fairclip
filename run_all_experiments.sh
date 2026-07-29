#!/usr/bin/env bash
# =============================================================================
# FairCLIP — Full multi-seed experiment runner
# 3 attributes (gender/age/race) x 3 seeds (42/123/456) = 9 training runs
# Each run: 30 epochs, patience 8, correct lambda per attribute.
# After each training, it evaluates and saves a per-seed result CSV.
# =============================================================================
set -e
cd /home/ayeshakalsoom/Downloads/thesis
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BACKBONE="ViT-B/32"
BB_TAG="ViT-B_32"
SEEDS=(42 123 456)

# lambda per attribute (gender already fair -> tiny; race most biased -> largest)
declare -A LAMBDA=( ["gender"]=0.01 ["age"]=0.5 ["race"]=1.0 )

for attr in gender age race; do
  for seed in "${SEEDS[@]}"; do
    echo "=============================================================="
    echo ">>> TRAINING  attr=$attr  seed=$seed  lambda=${LAMBDA[$attr]}"
    echo "=============================================================="
    rm -rf "results/checkpoints/${BB_TAG}/${attr}/seed${seed}"
    python -m training.train \
        --attribute "$attr" \
        --backbone "$BACKBONE" \
        --epochs 30 \
        --patience 8 \
        --batch_size 64 \
        --lambda_fair "${LAMBDA[$attr]}" \
        --seed "$seed"

    echo ">>> EVALUATING attr=$attr seed=$seed"
    python -m evaluation.eval_all \
        --model_path "results/checkpoints/${BB_TAG}/${attr}/seed${seed}/best_model.pt" \
        --backbone "$BACKBONE" \
        --attribute "$attr"

    # rename result CSV so each seed is kept separately
    if [ -f "results/tables/eval_FairCLIP_${BB_TAG}_${attr}.csv" ]; then
        mv "results/tables/eval_FairCLIP_${BB_TAG}_${attr}.csv" \
           "results/tables/eval_FairCLIP_${BB_TAG}_${attr}_seed${seed}.csv"
    fi
  done
done

echo "=============================================================="
echo ">>> ALL 9 RUNS COMPLETE. Per-seed CSVs in results/tables/"
echo "=============================================================="
