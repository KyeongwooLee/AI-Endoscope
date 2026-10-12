#!/bin/bash
set -euo pipefail
cd /tmp/gsm8k-train-all-v1
exec 9>resume.lock
flock -w 30 9
export HF_HOME=/tmp/gsm8k-confirmation-v1/cache
export HF_HUB_DISABLE_XET=1
export MPLCONFIGDIR=/tmp/gsm8k-train-all-v1/mpl
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHON=/tmp/gsm8k-confirmation-v1/env/bin/python
"$PYTHON" -u prefetch_train_all_model.py
"$PYTHON" -u verify_train_all_restart.py
echo "EXTRACTION_RESUMED $(date -u +%FT%TZ)"
"$PYTHON" -u extract_train_all.py --root /tmp/gsm8k-train-all-v1
echo "ALL_STAGES_COMPLETE $(date -u +%FT%TZ)"
