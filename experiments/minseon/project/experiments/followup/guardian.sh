#!/usr/bin/env bash
# Detached continuity wrapper: waits for the existing runner; never overlaps GPU jobs.
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=runs/20261012_followup
PYTHON=${PYTHON:-/tmp/yaicon-ab/venv/bin/python}
export HF_HOME=${HF_HOME:-/tmp/yaicon-ab/huggingface}
echo "$$" > "$RUN/guardian.pid"
previous_pid=${1:-0}
if [[ "$previous_pid" != 0 ]]; then
    while kill -0 "$previous_pid" 2>/dev/null; do sleep 5; done
fi
if [[ ! -f "$RUN/test_results.json" || ! -f "$RUN/REPORT.md" ]]; then
    bash experiments/followup/run.sh
fi
if [[ ! -f "$RUN/deployment_benchmark.json" ]]; then
    "$PYTHON" experiments/followup/benchmark_apply.py
fi
"$PYTHON" experiments/followup/application_smoke.py
"$PYTHON" experiments/followup/audit.py
"$PYTHON" experiments/followup/report.py
"$PYTHON" experiments/followup/finalize.py
