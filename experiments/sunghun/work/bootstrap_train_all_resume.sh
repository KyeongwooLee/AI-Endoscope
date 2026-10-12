#!/bin/bash
set -euo pipefail
cd /tmp/gsm8k-train-all-v1
exec 9>resume.lock
flock -n 9 || { echo 'Another resume job already holds the lock'; exit 1; }
export HF_HOME=/tmp/gsm8k-confirmation-v1/cache
export HF_HUB_DISABLE_XET=1
export MPLCONFIGDIR=/tmp/gsm8k-train-all-v1/mpl
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TMPDIR=/tmp/gsm8k-train-all-v1/install-tmp
mkdir -p "$TMPDIR" /tmp/gsm8k-confirmation-v1
if [ ! -x /tmp/gsm8k-confirmation-v1/env/bin/python ]; then
    python -m venv /tmp/gsm8k-confirmation-v1/env
fi
PYTHON=/tmp/gsm8k-confirmation-v1/env/bin/python
echo "SETUP_STARTED $(date -u +%FT%TZ)"
"$PYTHON" -m pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
"$PYTHON" -m pip install --no-cache-dir transformers==4.57.6 datasets==4.4.1 huggingface-hub==0.36.0 safetensors==0.7.0 numpy==2.2.6 pandas==2.3.3 pyarrow==22.0.0 matplotlib==3.10.8 scikit-learn==1.7.2
"$PYTHON" -u verify_train_all_restart.py
echo "EXTRACTION_RESUMED $(date -u +%FT%TZ)"
"$PYTHON" -u extract_train_all.py --root /tmp/gsm8k-train-all-v1
echo "ALL_STAGES_COMPLETE $(date -u +%FT%TZ)"
