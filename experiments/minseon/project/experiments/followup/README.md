# Frozen-model GSM8K follow-up

This is an isolated follow-up to `runs/20261012_AB`. The old run is a read-only exploratory source. All fresh original/verification/repair inference uses the same frozen Qwen2.5-1.5B-Instruct revision in FP32. See `config.json` for the canonical settings and `runs/20261012_followup/REPORT.md` for completed results once final evaluation finishes.

Run from repository root:

```bash
export HF_HOME=/tmp/yaicon-ab/huggingface
export PYTHON=/tmp/yaicon-ab/venv/bin/python
bash experiments/followup/run.sh
```

The runner executes GPU phases sequentially, saves each generation and record incrementally, skips existing records, and refuses fresh-test collection before all classifiers and policies are frozen. Do not run two GPU collectors together. The cache/environment is temporary; package versions are pinned in `runs/20261012_AB/requirements-lock.txt`. No reference text is passed to the model or feature builder.

The 40-example pilot is separate from the 1,200-question training pool, fixed 200-question development set and 300-question fresh official-test subset. All previous local question hashes are excluded. Seeds101/202 define two permutations of the training pool; 200/500/1000 prefixes are nested within each permutation. Primary comparison is n1000, seed101; other settings are secondary stability analyses. Small classifiers and all preprocessing fit only their training subset. A and B share original responses, scoring, splits and external features. B runs independently of A test findings.

A uses four external baselines (confidence/length, +TF-IDF, +old verification controls, +answer-token confidence and both verifier-label scores). Internal candidates add one of three hidden positions at block14 using one of three predeclared representations. Each uses C=.01/.1/1. `scaled` reproduces the prior coordinate standardization divided by sqrt1536; `standard` removes block shrinkage; `pca32` fits non-whitened randomized PCA on training-standardized coordinates. Different relative feature norms imply different regularization tradeoffs, not an assumption that unshrunk hidden features will win.

B compares Keep, validation-best fixed action, detector-triggered fixed repair, four independent action-success heads, and conservative success margins. Unavailable Recalculate is masked and falls back to Keep for fixed/trigger policies. Repair actions are independent. The error detector is trained only on the training subset; thresholds/actions/margins are selected on held-out development, so no in-sample stacking occurs. Success heads are selected by validation log loss rather than the prior run's policy accuracy. No test results feed these choices.

`scoring.py` v3 adds reference-free answer spans and fixes demonstrated extraction cases. Legacy numerical labels and strict-format labels are retained in every fresh record for sensitivity. The original calculator (including its own extraction eligibility) is unchanged. Recovery/damage probes condition training/evaluation groups on correctness but never expose correctness to a deployed selector. Sparse groups are deferred explicitly.

Code:

- `diagnose.py`, `extra_diagnostics.py`: original-run exploratory diagnostics and audit sampling frame.
- `prepare.py`: deterministic isolated manifests/settings.
- `collect.py`: standard inference, numeric-token confidence, both-label scoring, causal hidden positions.
- `learn.py`: limited train/validation comparisons, freeze, then fresh-test evaluation and paired bootstrap.
- `audit.py`, `test_followup.py`: integrity, unavailable masking, scoring spans, input isolation.
- `report.py`: tables, learning curves, paired uncertainty and report.
- `benchmark_batch.py`: pilot-only resource check; its duplicated generations are excluded from scientific cohorts.

Raw references live in the manifest and outcome records for scoring/audit only. Prediction inputs are explicitly whitelisted. Hidden arrays store only three small vectors per question; no full-layer activations or vocabulary logits are retained. Source snapshots/hashes, old-run preservation hashes and frozen model hashes make stages reviewable. Experiment C, FinQA, larger models and steering are deferred.

A pre-fit v3 scoring amendment fixed training-only measurement regressions discovered during collection. The 312 intermediate records and prior scoring implementation are archived; the change ledger identifies all four changed repair labels and the one re-extracted original span. No original or repair response was regenerated, and no fresh-test outcome existed.
