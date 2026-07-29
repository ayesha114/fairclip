#!/usr/bin/env bash
# UTKFace multi-seed — RESUMABLE. Skips runs already done. Safe to re-run after crash.
cd /home/ayeshakalsoom/Downloads/thesis
source /home/ayeshakalsoom/Downloads/fairclip-thesis/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BB="ViT-B/32"; BBT="ViT-B_32"
declare -A LAMBDA=( ["gender"]=0.01 ["age"]=0.5 )

log(){ echo "[$(date '+%H:%M:%S')] $*"; }

for seed in 123 456; do
  for attr in gender age; do
    CKPT="results/checkpoints/utkface/${BBT}/${attr}/seed${seed}/best_model.pt"
    RETR="results/tables/retrieval_inlp_utkface_${attr}_seed${seed}.csv"

    if [ -f "$RETR" ]; then
      log "SKIP $attr seed=$seed (already complete)"; continue
    fi

    log ">>> TRAIN $attr seed=$seed"
    if [ ! -f "$CKPT" ]; then
      python -m training.train --dataset utkface --attribute "$attr" --backbone "$BB" \
          --epochs 30 --patience 8 --batch_size 64 --lambda_fair "${LAMBDA[$attr]}" --seed "$seed" \
          || { log "TRAIN FAILED $attr seed=$seed"; continue; }
    else
      log "  (checkpoint exists, skipping training)"
    fi

    log ">>> EVAL classification $attr seed=$seed"
    python -m evaluation.eval_all --dataset utkface --model_path "$CKPT" \
        --backbone "$BB" --attribute "$attr" \
        && mv "results/tables/eval_FairCLIP_${BBT}_utkface_${attr}.csv" \
              "results/tables/eval_FairCLIP_${BBT}_utkface_${attr}_seed${seed}.csv" 2>/dev/null || true

    log ">>> EVAL retrieval $attr seed=$seed"
    MI=15; [ "$attr" = "gender" ] && MI=30
    python eval_retrieval_inlp.py --dataset utkface --attribute "$attr" --seed "$seed" --max_iters $MI \
        || log "RETRIEVAL FAILED $attr seed=$seed"
  done
done
log ">>> UTKFACE MULTI-SEED COMPLETE"
