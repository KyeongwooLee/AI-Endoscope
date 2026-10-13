#!/usr/bin/env bash
set -euo pipefail
TASK_TMP=/tmp/yaicon-ab
export PIP_CACHE_DIR="$TASK_TMP/pip" HF_HOME="$TASK_TMP/huggingface"
mkdir -p "$TASK_TMP"
if [[ ! -x "$TASK_TMP/venv/bin/pip" ]]; then
  python3 -m pip --isolated install --target "$TASK_TMP/bootstrap" virtualenv
  PYTHONPATH="$TASK_TMP/bootstrap" python3 -m virtualenv --app-data "$TASK_TMP/virtualenv" "$TASK_TMP/venv"
fi
"$TASK_TMP/venv/bin/python" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
"$TASK_TMP/venv/bin/python" -m pip install transformers==4.57.1 datasets==4.3.0 scikit-learn==1.7.2 numpy==2.2.6 accelerate==1.10.1 matplotlib==3.10.7 pypdf

TASK_ROOT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
if [[ -f "$TASK_ROOT/runs/20261012_AB/requirements-lock.txt" ]]; then
  "$TASK_TMP/venv/bin/python" -m pip install -r "$TASK_ROOT/runs/20261012_AB/requirements-lock.txt" --extra-index-url https://download.pytorch.org/whl/cu126
fi
