#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PHASE="${1:-pilot}"
case "$PHASE" in pilot|exploration|test) ;; *) printf 'Use pilot, exploration, or test\n' >&2; exit 2 ;; esac
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CONFIG="${CONFIG:-configs/rtx3090.json}"
RUN_DIR="${RUN_DIR:-/tmp/gsm8k-study/run-v2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

"$PYTHON_BIN" -m gsm8k_lens prepare --config "$CONFIG" --run "$RUN_DIR"
"$PYTHON_BIN" -m gsm8k_lens run-batch --config "$CONFIG" --run "$RUN_DIR" --phase "$PHASE" --batch-size "${BATCH_SIZE:-128}"
printf '\nOne bounded batch is ready. Transfer and verify locally before acknowledging cleanup.\n'
printf 'For automatic pull/verify/ack, run scripts/sync_from_vessl.py on the LOCAL computer.\n'
