#!/usr/bin/env bash
cd /home/ayeshakalsoom/Downloads/thesis
source /home/ayeshakalsoom/Downloads/fairclip-thesis/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=rerun_all_final.log
echo "START $(date)" > $LOG
for bb in "ViT-B/32:ViT-B_32" "ViT-B/16:ViT-B_16" "ViT-L/14:ViT-L_14" "ViT-H/14:ViT-H_14"; do
  bbn="${bb%%:*}"; bbt="${bb##*:}"
  for attr in gender age race; do
    # baseline
    echo "BASELINE: $bbt $attr $(date)" | tee -a $LOG
    python -m evaluation.eval_all --dataset fairface --baseline \
      --backbone "$bbn" --attribute "$attr" > "final_baseline_${bbt}_${attr}.log" 2>&1
    # fairclip
    ck="results/checkpoints/fairface/${bbt}/${attr}/seed42/best_model.pt"
    if [ -f "$ck" ]; then
      echo "FAIRCLIP: $bbt $attr $(date)" | tee -a $LOG
      python -m evaluation.eval_all --dataset fairface --model_path "$ck" \
        --backbone "$bbn" --attribute "$attr" > "final_fairclip_${bbt}_${attr}.log" 2>&1
    fi
  done
done
echo "ALL DONE $(date)" | tee -a $LOG
echo done > rerun_all_final_complete.flag
