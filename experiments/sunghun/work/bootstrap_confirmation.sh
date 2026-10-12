#!/bin/bash
set -euo pipefail
cd /tmp/gsm8k-confirmation-v1
export HF_HOME=/tmp/gsm8k-confirmation-v1/cache
export HF_HUB_DISABLE_XET=1
export MPLCONFIGDIR=/tmp/gsm8k-confirmation-v1/mpl
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m venv env
env/bin/python -m pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
env/bin/python -m pip install --no-cache-dir transformers==4.57.6 datasets==4.4.1 huggingface-hub==0.36.0 safetensors==0.7.0 numpy==2.2.6 pandas==2.3.3 pyarrow==22.0.0 matplotlib==3.10.8 scikit-learn==1.7.2
env/bin/python -u run_confirmation.py
echo ALL_STAGES_COMPLETE
