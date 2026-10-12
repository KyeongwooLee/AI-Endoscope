#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export GSM8K_SCRATCH="${GSM8K_SCRATCH:-/tmp/gsm8k-study}"
export TMPDIR="${TMPDIR:-$GSM8K_SCRATCH/tmp}"
export HF_HOME="${HF_HOME:-$GSM8K_SCRATCH/hf-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$GSM8K_SCRATCH/xdg-cache}"
export PIP_NO_CACHE_DIR=1
mkdir -p "$TMPDIR" "$HF_HOME" "$XDG_CACHE_HOME"
df -h "$GSM8K_SCRATCH"
df -i "$GSM8K_SCRATCH"
findmnt -T "$GSM8K_SCRATCH"

# Use a CUDA-capable Linux host with Python 3.10 through 3.12 and an NVIDIA driver.
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), "Use Python 3.10 through 3.12"'
"$PYTHON_BIN" -m venv --without-pip .venv
"$PYTHON_BIN" -m pip --python .venv/bin/python install --upgrade pip
.venv/bin/python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m gsm8k_lens doctor --config configs/rtx3090.json --run "$GSM8K_SCRATCH/run-v2"
printf '\nSetup complete. Activate with: source .venv/bin/activate\n'
