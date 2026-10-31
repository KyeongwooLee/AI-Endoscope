#!/bin/bash
# GSM8K self-generation on 2 GPUs (problem-sharded): greedy + 4 samples per problem
cd "$(dirname "$0")"
export HF_HOME=/root/hfhome
echo "[gen start] $(date)" > status.txt
for s in 0 1; do
  CUDA_VISIBLE_DEVICES=$s ../.venv-vllm/bin/python generate.py ../data/gsm8k gen 4 $s 2 > gen_$s.log 2>&1 &
done
wait
grep -h -E "^(test|train):" gen_*.log >> status.txt
echo "[gen done] $(date)" >> status.txt
