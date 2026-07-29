#!/usr/bin/env bash
cd /home/ayeshakalsoom/Downloads/thesis
source /home/ayeshakalsoom/Downloads/fairclip-thesis/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
./run_final3.sh >> final3_log.txt 2>&1
./run_age_fix.sh >> agefix_log.txt 2>&1
echo done > tonight_done.txt
