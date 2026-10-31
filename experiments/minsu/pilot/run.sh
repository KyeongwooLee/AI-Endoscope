#!/bin/bash
# EIC pilot: feature extraction (Qwen2.5-1.5B-Instruct) -> probes. Results in results.txt
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
export HF_HOME=/root/hfhome
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "[start] $(date)" > status.txt
python extract.py data/EIC feat Qwen/Qwen2.5-1.5B-Instruct > extract.log 2>&1
echo "[extract done] $(date)" >> status.txt
python probe.py feat data/gsm8k > results.txt 2> probe.err
echo "[probe done] $(date)" >> status.txt
