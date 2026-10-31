#!/bin/bash
# Pilot 2: step-level features for all steps -> first-wrong-step localization without oracle position.
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
export HF_HOME=/root/hfhome
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "[start] $(date)" > status_steps.txt
python extract_steps.py data/EIC feat_steps Qwen/Qwen2.5-1.5B-Instruct > extract_steps.log 2>&1
echo "[extract done] $(date)" >> status_steps.txt
python probe_steps.py feat_steps > results_steps.txt 2> probe_steps.err
echo "[probe done] $(date)" >> status_steps.txt
