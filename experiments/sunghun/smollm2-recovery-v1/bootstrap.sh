#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/ai-endoscope/smollm2-recovery-v1
export PIP_CACHE_DIR=/tmp/smollm2-recovery-pip
export TMPDIR=/tmp/smollm2-recovery-tmp
mkdir -p "$ROOT" "$TMPDIR"
python3 -m venv --system-site-packages "$ROOT/venv"
"$ROOT/venv/bin/python" -m pip install --disable-pip-version-check 'transformers==4.46.3' 'huggingface-hub==0.36.0' 'safetensors==0.6.2' 'numpy==1.26.4'
"$ROOT/venv/bin/python" "$ROOT/src/launch.py" --root "$ROOT"
