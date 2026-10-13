# Reproduce the saved A/B experiments

> Publication status: code, reports, figures and summary results are ready for upload. Raw response/feature/model bundles are prepared locally but are NOT uploaded; public export requires additional authorization after automatic approval review blocked it. Data restoration instructions below apply once those bundles are published.

## Restore and inspect (no GPU)

```bash
git clone --branch minseon --single-branch https://github.com/KyeongwooLee/AI-Endoscope.git
cd AI-Endoscope/experiments/minseon
python3 restore_data.py --verify-only
python3 restore_data.py
cd project
```

The `project/experiments` and `project/runs` layout is intentional: original sources resolve their project root relative to `__file__`. Do not move just one script out of that layout.

## Environment and CPU checks

Recorded environment: Python 3.10, torch 2.8.0+cu126, transformers 4.57.1, datasets 4.3.0, NumPy 2.2.6, scikit-learn 1.7.2, matplotlib 3.10.7. Full pins: `runs/20261012_AB/requirements-lock.txt` and run-specific `packages.json`.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r runs/20261012_AB/requirements-lock.txt
python experiments/ab/test_invariants.py
python experiments/followup/test_followup.py
python experiments/followup/apply.py \
  --inputs runs/20261012_followup/application_smoke_inputs.json \
  --hidden-dir runs/20261012_followup/hidden \
  --output /tmp/minseon-application-predictions.json
```

The inference command uses only whitelisted pre-repair features and frozen small models. It does not need the Qwen weights or a GPU. Saved smoke checks document exact agreement with evaluation.

## Recompute evaluation from frozen artifacts

Use a disposable restored copy if preserving downloaded bytes/hashes matters. The following commands write refreshed metrics, timing values and report artifacts:

```bash
python experiments/followup/learn.py evaluate
python experiments/followup/report.py
```

Do not refit the saved run or select new settings from the published test. Models and decisions were frozen before its fresh-test collection. `learn.py fit` intentionally refuses an already frozen directory. The explicit publication exclusions mean historical all-file preservation audits expect source documents that are not in this public package; `restore_data.py --verify-only` is the appropriate publication integrity check.

## GPU collection / archived execution

The recorded main follow-up used one GTX 1080 Ti, FP32, sequential GPU jobs, batch16 originals and batch32 mixed repairs. Prompts, decoding, revisions, exact split IDs and selected settings are saved. `experiments/followup/run.sh` is the archived resumable orchestration script; it was not rerun during publication. It assumes the recorded run layout and defaults to `/tmp/yaicon-ab/venv/bin/python` and `/tmp/yaicon-ab/huggingface`; override `PYTHON` and `HF_HOME` for another machine. The one-time scoring amendment script is historical and must not be rerun on the completed artifacts.

For new inference, use a new isolated run and deliberately update its run paths/configuration. Do not claim an exact historical replay if hardware, precision, decoding, scoring, split IDs or package versions differ. The saved reports separate pilot observations, original exploratory findings and the fresh follow-up evaluation. No new GPU inference or scientific retuning was performed to publish this branch.
