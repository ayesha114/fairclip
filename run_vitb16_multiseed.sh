#!/usr/bin/env bash
# ViT-B/16 FairFace multi-seed — RESUMABLE. Skips completed runs.
cd /home/ayeshakalsoom/Downloads/thesis
source /home/ayeshakalsoom/Downloads/fairclip-thesis/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BB="ViT-B/16"; BBT="ViT-B_16"
declare -A LAMBDA=( ["gender"]=0.01 ["age"]=0.5 ["race"]=1.0 )
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

for seed in 42 123 456; do
  for attr in gender age race; do
    CKPT="results/checkpoints/fairface/${BBT}/${attr}/seed${seed}/best_model.pt"
    RETR="results/tables/retrieval_inlp_vitb16_${attr}_seed${seed}.csv"
    if [ -f "$RETR" ]; then log "SKIP $attr seed=$seed (done)"; continue; fi

    log ">>> TRAIN $attr seed=$seed"
    if [ ! -f "$CKPT" ]; then
      python -m training.train --dataset fairface --attribute "$attr" --backbone "$BB" \
          --epochs 30 --patience 8 --batch_size 64 --lambda_fair "${LAMBDA[$attr]}" --seed "$seed" \
          || { log "TRAIN FAILED $attr seed=$seed"; continue; }
    fi

    log ">>> EVAL classification $attr seed=$seed"
    python -m evaluation.eval_all --dataset fairface --model_path "$CKPT" \
        --backbone "$BB" --attribute "$attr" \
        && mv "results/tables/eval_FairCLIP_${BBT}_${attr}.csv" \
              "results/tables/eval_FairCLIP_${BBT}_${attr}_seed${seed}.csv" 2>/dev/null || true

    log ">>> EVAL retrieval $attr seed=$seed"
    if [ "$attr" = "gender" ]; then MI=30; else MI=20; fi
    python eval_retrieval_inlp.py --dataset fairface --attribute "$attr" --seed "$seed" \
        --backbone "$BB" --max_iters "$MI" \
        || log "RETRIEVAL FAILED $attr seed=$seed"
    mv "results/tables/retrieval_inlp_${attr}_seed${seed}.csv" "$RETR" 2>/dev/null || true
  done
done
log ">>> ViT-B/16 MULTI-SEED COMPLETE"
