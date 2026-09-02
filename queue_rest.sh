#!/usr/bin/env bash
cd /home/ayeshakalsoom/Downloads/thesis
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f ff_multiseed.sh > /dev/null; do sleep 60; done
python complete_metrics_seed.py 123
python complete_metrics_seed.py 456
echo MULTISEED_METRICS_DONE
