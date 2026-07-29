#!/usr/bin/env bash
# ViT-H/14 FairFace single-seed (42) — OpenCLIP, FP16, RESUMABLE
cd /home/ayeshakalsoom/Downloads/thesis
source /home/ayeshakalsoom/Downloads/fairclip-thesis/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BB="ViT-H/14"; BBT="ViT-H_14"; seed=42
declare -A LAMBDA=( ["gender"]=0.01 ["age"]=0.5 ["race"]=1.0 )
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

for attr in gender age race; do
  CKPT="results/checkpoints/fairface/${BBT}/${attr}/seed${seed}/best_model.pt"
  RETR="results/tables/retrieval_inlp_vith14_${attr}_seed${seed}.csv"
  if [ -f "$RETR" ]; then log "SKIP $attr (done)"; continue; fi
  log ">>> TRAIN $attr"
  if [ ! -f "$CKPT" ]; then
    python -m training.train --dataset fairface --attribute "$attr" --backbone "$BB" \
        --epochs 30 --patience 8 --batch_size 16 --lambda_fair "${LAMBDA[$attr]}" --seed "$seed" \
        || { log "TRAIN FAILED $attr"; continue; }
  fi
  log ">>> EVAL classification $attr"
  python -m evaluation.eval_all --dataset fairface --model_path "$CKPT" \
      --backbone "$BB" --attribute "$attr" \
      && mv "results/tables/eval_FairCLIP_${BBT}_${attr}.csv" \
            "results/tables/eval_FairCLIP_${BBT}_${attr}_seed${seed}.csv" 2>/dev/null || true
  log ">>> EVAL retrieval $attr"
  if [ "$attr" = "gender" ]; then MI=50; else MI=20; fi
  python eval_retrieval_inlp.py --dataset fairface --attribute "$attr" --seed "$seed" \
      --backbone "$BB" --max_iters "$MI" || log "RETR FAILED $attr"
  mv "results/tables/retrieval_inlp_${attr}_seed${seed}.csv" "$RETR" 2>/dev/null || true
done
log ">>> ViT-H/14 COMPLETE"
