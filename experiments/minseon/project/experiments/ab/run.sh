#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
PY=/tmp/yaicon-ab/venv/bin/python
RUN=runs/20261012_AB
export HF_HOME=/tmp/yaicon-ab/huggingface OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p "$RUN"
"$PY" experiments/ab/prepare_run.py
"$PY" experiments/ab/pipeline.py collect --split pilot
"$PY" experiments/ab/pipeline.py audit
"$PY" experiments/ab/pipeline.py collect --split train
"$PY" experiments/ab/pipeline.py collect --split validation
if [[ ! -f "$RUN/selection_frozen.json" ]]; then
  "$PY" experiments/ab/evaluate.py fit
fi
"$PY" experiments/ab/sanity.py
"$PY" experiments/ab/pipeline.py collect --split test
"$PY" experiments/ab/evaluate.py evaluate

"$PY" experiments/ab/audit.py
if [[ ! -f "$RUN/deployment_benchmark.json" ]]; then
  "$PY" experiments/ab/benchmark_apply.py
fi
"$PY" experiments/ab/export_results.py
"$PY" experiments/ab/reference_sensitivity.py
"$PY" experiments/ab/report.py
"$PY" experiments/ab/finalize.py
