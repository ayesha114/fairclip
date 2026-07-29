#!/usr/bin/env bash
cd /home/ayeshakalsoom/Downloads/thesis
source /home/ayeshakalsoom/Downloads/fairclip-thesis/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run_abl () {
  NAME=$1; shift
  python -m training.train --dataset fairface --attribute race --backbone ViT-B/32 \
    --epochs 30 --patience 8 --batch_size 64 --lambda_fair 2.0 --n_bias_dirs 5 --seed 42 \
    --output_dir "results/checkpoints_abl_${NAME}" "$@" > "abl_${NAME}.txt" 2>&1
  CKPT=$(find "results/checkpoints_abl_${NAME}" -name best_model.pt | head -1)
  [ -n "$CKPT" ] && python -m evaluation.eval_all --dataset fairface --model_path "$CKPT" \
    --backbone ViT-B/32 --attribute race > "abl_eval_${NAME}.txt" 2>&1
}
rm -rf results/checkpoints_abl_no_trainproj
run_abl no_trainproj --no_train_proj
run_abl no_procrustes --no_procrustes
run_abl no_subspace --no_subspace
for NAME in no_fairloss no_temp; do
  [ -f "abl_eval_${NAME}.txt" ] && continue
  CKPT=$(find "results/checkpoints_abl_${NAME}" -name best_model.pt | head -1)
  [ -n "$CKPT" ] && python -m evaluation.eval_all --dataset fairface --model_path "$CKPT" \
    --backbone ViT-B/32 --attribute race > "abl_eval_${NAME}.txt" 2>&1
done
echo done > ablations_done.txt
