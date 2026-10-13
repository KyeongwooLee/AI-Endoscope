#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON=${PYTHON:-/tmp/yaicon-ab/venv/bin/python}
export HF_HOME=${HF_HOME:-/tmp/yaicon-ab/huggingface}
while pgrep -f '^/tmp/yaicon-ab/venv/bin/python experiments/followup/collect.py train$' >/dev/null; do sleep 5; done
"$PYTHON" - <<'PY'
import json
from pathlib import Path
p=Path('runs/20261012_followup');m=json.loads((p/'manifest.json').read_text());assert all((p/'records'/f"{r['id']}.json").exists() for r in m if r['split']=='train'),'Training collection incomplete: resume manually'
PY
"$PYTHON" experiments/followup/collect.py validation
"$PYTHON" experiments/followup/audit.py
if [[ ! -f runs/20261012_followup/selection_frozen.json ]]; then
    "$PYTHON" experiments/followup/learn.py fit
fi
"$PYTHON" experiments/followup/collect.py test
"$PYTHON" experiments/followup/learn.py evaluate
"$PYTHON" experiments/followup/audit.py
"$PYTHON" experiments/followup/report.py
