#!/bin/bash
# waits for generation, then feature extraction on 2 GPUs, then solution-level probes -> results.txt
cd "$(dirname "$0")"
until grep -q "gen done" status.txt; do sleep 20; done
source ../.venv/bin/activate
export HF_HOME=/root/hfhome
echo "[extract start] $(date)" >> status.txt
for s in 0 1; do
  CUDA_VISIBLE_DEVICES=$s python extract_gen.py gen feat $s 2 > extract_$s.log 2>&1 &
done
wait
echo "[extract done] $(date)" >> status.txt
python probe_gen.py feat > results.txt 2> probe.err
echo "[probe done] $(date)" >> status.txt
