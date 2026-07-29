#!/usr/bin/env bash
cd /home/ayeshakalsoom/Downloads/thesis
source /home/ayeshakalsoom/Downloads/fairclip-thesis/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- STAGE 1: utility re-runs (ImageNetV2 + Flickr, now fixed) ----
rm -f results/tables/flickr_utility.csv flickr_log.txt
for bb in "ViT-B/32" "ViT-B/16" "ViT-L/14"; do
  bbt=$(echo "$bb" | tr '/' '_')
  python eval_imagenetv2.py --baseline --backbone "$bb" > "util_inv2_${bbt}_baseline.txt" 2>&1
  python eval_flickr.py --baseline --backbone "$bb" --tag "baseline_${bbt}" >> flickr_log.txt 2>&1
  for attr in gender age race; do
    CKPT="results/checkpoints/fairface/${bbt}/${attr}/seed42/best_model.pt"
    [ -f "$CKPT" ] || continue
    python eval_imagenetv2.py --model_path "$CKPT" --backbone "$bb" > "util_inv2_${bbt}_${attr}.txt" 2>&1
    python eval_flickr.py --model_path "$CKPT" --backbone "$bb" --tag "fairclip_${bbt}_${attr}" >> flickr_log.txt 2>&1
  done
done
echo "STAGE1 utility done"

# ---- STAGE 2: race lever test (75% COCO + train-time projection) ----
python3 - << 'PY'
f="training/train.py"; s=open(f).read()
s=s.replace('texts = [prompts[min(l.item(), len(prompts)-1)] if i % 2 == 0',
            'texts = [prompts[min(l.item(), len(prompts)-1)] if i % 4 == 0')
open(f,"w").write(s); print("ratio -> 25% prompts / 75% COCO")
PY
rm -rf results/checkpoints/fairface/ViT-B_32/race/seed42_lever
python -m training.train --dataset fairface --attribute race --backbone ViT-B/32 \
  --epochs 30 --patience 8 --batch_size 64 --lambda_fair 2.0 --n_bias_dirs 5 --seed 999 \
  > race_lever75.txt 2>&1
python -m evaluation.eval_all --dataset fairface \
  --model_path results/checkpoints/fairface/ViT-B_32/race/seed999/best_model.pt \
  --backbone ViT-B/32 --attribute race > race_lever75_eval.txt 2>&1
# restore 50/50 for future runs
python3 - << 'PY'
f="training/train.py"; s=open(f).read()
s=s.replace('texts = [prompts[min(l.item(), len(prompts)-1)] if i % 4 == 0',
            'texts = [prompts[min(l.item(), len(prompts)-1)] if i % 2 == 0')
open(f,"w").write(s); print("ratio restored -> 50/50")
PY
echo "STAGE2 lever done"

# ---- STAGE 3: UTKFace (locked recipe, gender+age+race) ----
declare -A LAMBDA=( ["gender"]=0.01 ["age"]=0.5 ["race"]=2.0 )
for attr in gender age race; do
  rm -rf results/checkpoints/utkface/ViT-B_32/${attr}/seed42
  python -m training.train --dataset utkface --attribute "$attr" --backbone ViT-B/32 \
    --epochs 30 --patience 8 --batch_size 64 --lambda_fair "${LAMBDA[$attr]}" \
    --n_bias_dirs 5 --seed 42 > "utk_${attr}.txt" 2>&1
  python -m evaluation.eval_all --dataset utkface \
    --model_path "results/checkpoints/utkface/ViT-B_32/${attr}/seed42/best_model.pt" \
    --backbone ViT-B/32 --attribute "$attr" > "utk_eval_${attr}.txt" 2>&1
done
echo night_done > night_done.txt
