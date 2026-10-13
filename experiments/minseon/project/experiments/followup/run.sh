#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME=${HF_HOME:-/tmp/yaicon-ab/huggingface}
PYTHON=${PYTHON:-/tmp/yaicon-ab/venv/bin/python}
RUN=runs/20261012_followup
if [[ ! -f "$RUN/original_preservation.json" ]]; then
    "$PYTHON" experiments/followup/diagnose.py
fi
"$PYTHON" experiments/followup/prepare.py
"$PYTHON" experiments/followup/test_followup.py
"$PYTHON" experiments/followup/collect.py pilot
"$PYTHON" experiments/followup/collect.py train
"$PYTHON" experiments/followup/collect.py validation
"$PYTHON" experiments/followup/audit.py
if [[ ! -f "$RUN/selection_frozen.json" ]]; then
    "$PYTHON" experiments/followup/learn.py fit
fi
"$PYTHON" experiments/followup/collect.py test
"$PYTHON" experiments/followup/learn.py evaluate
"$PYTHON" experiments/followup/audit.py
"$PYTHON" experiments/followup/report.py
if [[ ! -f "$RUN/deployment_benchmark.json" ]]; then
    "$PYTHON" experiments/followup/benchmark_apply.py
fi
"$PYTHON" experiments/followup/application_smoke.py
"$PYTHON" experiments/followup/report.py
"$PYTHON" experiments/followup/finalize.py
