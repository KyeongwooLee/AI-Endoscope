#!/usr/bin/env bash
set -euo pipefail
ROOT=/tmp/smollm2-detailed-v1
mkdir -p "$ROOT"
export PIP_CACHE_DIR="$ROOT/pip-cache"
export HF_HOME="$ROOT/cache"
export HF_HUB_DISABLE_XET=1
export TMPDIR="$ROOT/tmp"
mkdir -p "$TMPDIR"
python3 -m venv --system-site-packages "$ROOT/venv"
"$ROOT/venv/bin/python" -m pip install --disable-pip-version-check 'transformers==4.46.3' 'huggingface-hub==0.36.0' 'safetensors==0.6.2' 'numpy==1.26.4'
"$ROOT/venv/bin/python" -c 'import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())'
