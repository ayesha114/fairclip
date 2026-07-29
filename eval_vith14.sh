#!/usr/bin/env bash
# ViT-H/14 evals — one at a time, GPU cleared between each (avoids OOM)
cd /home/ayeshakalsoom/Downloads/thesis
source /home/ayeshakalsoom/Downloads/fairclip-thesis/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BB="ViT-H/14"; BBT="ViT-H_14"; seed=42
log(){ echo "[$(date '+%H:%M:%S')] $*"; }

for attr in gender age race; do
  CKPT="results/checkpoints/fairface/${BBT}/${attr}/seed${seed}/best_model.pt"

  # classification (skip if done)
  CLS="results/tables/eval_FairCLIP_${BBT}_${attr}_seed${seed}.csv"
  if [ ! -f "$CLS" ]; then
    log ">>> classification $attr"
    python -m evaluation.eval_all --dataset fairface --model_path "$CKPT" \
        --backbone "$BB" --attribute "$attr" \
        && mv "results/tables/eval_FairCLIP_${BBT}_${attr}.csv" "$CLS" 2>/dev/null
    sleep 5   # let GPU clear
  else
    log "skip classification $attr (done)"
  fi

  # retrieval (skip if done)
  RETR="results/tables/retrieval_inlp_vith14_${attr}_seed${seed}.csv"
  if [ ! -f "$RETR" ]; then
    log ">>> retrieval $attr"
    if [ "$attr" = "gender" ]; then MI=50; else MI=20; fi
    python eval_retrieval_inlp.py --dataset fairface --attribute "$attr" --seed "$seed" \
        --backbone "$BB" --max_iters "$MI" \
        && mv "results/tables/retrieval_inlp_${attr}_seed${seed}.csv" "$RETR" 2>/dev/null
    sleep 5
  else
    log "skip retrieval $attr (done)"
  fi
done
log ">>> ViT-H/14 EVAL COMPLETE"
