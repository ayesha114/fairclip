#!/usr/bin/env bash
cd /home/ayeshakalsoom/Downloads/thesis
source /home/ayeshakalsoom/Downloads/fairclip-thesis/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[1/3] round selection + H14 race iterative"
python3 round_select.py > round_tables.txt 2>&1

echo "[2/3] no_subspace ablation"
python -m training.train --dataset fairface --attribute race --backbone ViT-B/32 \
  --epochs 30 --patience 3 --batch_size 32 --lambda_fair 2.0 --n_bias_dirs 5 --seed 42 \
  --no_subspace --no_train_proj --output_dir results/checkpoints_abl_nosub3 > abl_nosub3.txt 2>&1
CK=$(find results/checkpoints_abl_nosub3 -name best_model.pt 2>/dev/null | head -1)
[ -n "$CK" ] && python -m evaluation.eval_all --dataset fairface --model_path "$CK" \
  --backbone ViT-B/32 --attribute race > abl_eval_nosub3.txt 2>&1

echo "[3/3] age lambda fix"
for LAM in 0.25 0.75; do
  OUT="results/checkpoints_agefix_lam${LAM}"
  rm -rf "$OUT"
  python -m training.train --dataset fairface --attribute age --backbone ViT-B/32 \
    --epochs 30 --patience 5 --batch_size 64 --lambda_fair $LAM --n_bias_dirs 5 --seed 42 \
    --output_dir "$OUT" > "agefix_lam${LAM}.txt" 2>&1
  CK=$(find "$OUT" -name best_model.pt 2>/dev/null | head -1)
  [ -n "$CK" ] && python -m evaluation.eval_all --dataset fairface --model_path "$CK" \
    --backbone ViT-B/32 --attribute age > "agefix_eval_lam${LAM}.txt" 2>&1
done
python3 agefix_able.py > agefix_able.txt 2>&1
echo done > tonight_done.txt
