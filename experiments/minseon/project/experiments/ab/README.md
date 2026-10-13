# GSM8K Experiments A and B

Code for the 2026-10-12 specification; C is deferred. Outputs are in `runs/20261012_AB`; earlier runs are retained unchanged. The source specification is saved as `source_notion.txt`. This directory is not a Git checkout.

## Execute or resume

```bash
bash experiments/ab/setup.sh
bash experiments/ab/run.sh > runs/20261012_AB/resume.log 2>&1
```

Use `/tmp/yaicon-ab/venv/bin/python` explicitly. The old temporary environment no longer existed; the current host has Python 3.10 and a GTX 1080 Ti, not the GPU recorded by prior experiments. Large weights and environment files stay under `/tmp`. `setup.sh` rebuilds them after a restart. Saved model and dataset commit hashes in `config.json` are used for all downloads. The script skips completed original, verification, and action stages and refuses to refit frozen settings. For a different experiment, use a new run directory and manifest. Do not edit settings and resume incompatible artifacts.

Individual commands:

```bash
/tmp/yaicon-ab/venv/bin/python experiments/ab/prepare_run.py
/tmp/yaicon-ab/venv/bin/python experiments/ab/pipeline.py collect --split pilot
/tmp/yaicon-ab/venv/bin/python experiments/ab/pipeline.py audit
/tmp/yaicon-ab/venv/bin/python experiments/ab/pipeline.py collect --split train
/tmp/yaicon-ab/venv/bin/python experiments/ab/pipeline.py collect --split validation
/tmp/yaicon-ab/venv/bin/python experiments/ab/evaluate.py fit
/tmp/yaicon-ab/venv/bin/python experiments/ab/pipeline.py collect --split test
/tmp/yaicon-ab/venv/bin/python experiments/ab/evaluate.py evaluate
/tmp/yaicon-ab/venv/bin/python -m unittest discover -s experiments/ab -p 'test_*.py'
```

## Protocol

40 pilot examples are excluded from training and evaluation. Main scale is fixed in `manifest.json`: 200 training and 80 validation examples from official GSM8K train, 100 final evaluation examples from official test. Training and validation retain the prior run's assignments and original-response artifacts. The previous run's train-derived test is not used. The official test sample is drawn with seed 20261012 before collecting outcomes; reused pilot/train/validation assignments inherit seed 42. Classifier seed is 20261012, bootstrap seed 20261013, shuffle seeds 9000–9019. Normalized question hashes prevent exact duplicates crossing splits; GSM8K supplies no richer family/source grouping, so undetected paraphrase families remain a limitation.

The model is frozen Qwen2.5-1.5B-Instruct, greedy decoding with inherited repetition_penalty=1.1 (also present in reused originals). Teacher-forced confidence uses raw model probabilities, not repetition-penalized decoding scores. `generation_effective.json` records inherited defaults and active overrides. This matches the team's model and greedy setting. Their meeting notes used 1,024 answer tokens, a different `####` prompt, 800/200 training/validation examples and the full 1,319 test examples. Here the prior local 512-token original prompt is retained to reuse original responses. New inference is FP32 because the current Pascal GPU lacks efficient FP16 arithmetic; reused originals were generated in FP16 on a V100. This precision difference may affect train/test comparability and is not a controlled team replication.

Original responses are generated once or reused, with original token IDs and generation confidence. All A/B comparisons use exactly the same examples and features. Hidden vectors from the prior run cannot substitute for the new verification-context positions, so they are recollected. References are stored for labeling only and never included in model prompts, text vectors, confidence, hidden-state inputs, or selection. C can reuse question IDs, original token IDs, response text, probabilities, and original verification-context features, but future C variants must retain the same question-group assignment.

Primary numeric extraction prefers explicit final markers, then numeric boxed answers, then a numeric complement of a concluding verb on the final nonempty line, then the last number. A truncated response without an explicit final numeric marker or box is a failure. Exact Decimal equality normalizes commas, dollars, signs, and decimal representations; percentages remain in the question's numeric units. No reference-based rounding or extraction is used. Last-number extraction can still confuse an intermediate number for an answer in untruncated responses. A separate strict-format sensitivity analysis quantifies dependence on parsing without changing selected models.

A compares D1 (mean generation token NLL and log response length), D2 (D1 plus train-fit TF-IDF question/response text), D3 (D2 plus exact explicit verifier judgment and verdict token NLL), and D4 (D3 plus one hidden-state vector). Invalid verifier outputs have a separate feature. Text is a modest lexical baseline, not a claim to represent all external detectors. Confidence uses mean raw-model NLL; richer prior numeric-span confidence and entropy controls are not included in this A/B comparison.

The verification prompt includes the full original response. Three token positions are recorded: the last token overlapping the completed response, the dedicated two-newline separator token, and the final chat prompt token immediately before the verdict. Segmented encoding prevents BPE merging response punctuation with the separator. The concatenated IDs decode exactly to the intended prompt and are used both for extraction and verdict generation. All vectors come from raw post-block hooks at blocks 7, 14, 21, 28, before final normalization. Pilot prefix checks independently confirm the positions are causal. The final response token may be punctuation; this is the completed-response representation, not necessarily the numeric-answer token.

The classifier is regularized logistic regression. TF-IDF, coordinate scalers, and hidden scaling are fit on training examples only. Hidden blocks are divided by sqrt(hidden width) after coordinate standardization. A selects C and hidden position/layer by validation AUROC; B independently selects by validation final accuracy, breaking ties by mean action log loss. Neither uses A test results to tune B. Both settings are serialized and hashed before final testing.

B trains four independent success classifiers. Unavailable calculator labels are missing, excluded from that classifier's fit, and masked during selection. Keep is exactly one minus A's error label. Multiple actions can succeed; probabilities do not sum to one. Recalculate safely executes a numeric equality from the model's own response only when its RHS matches the model's extracted final answer. It does not infer symbolic dependencies or consult references. Rebuild sees the original question alone; Review sees question and original response. All actions start independently. Fixed Recalculate falls back to Keep when unavailable. Ties in selection prefer Keep, Recalculate, Rebuild, Review in that order. The best fixed policy and best fixed repair are selected on validation.

## Outputs and safeguards

`stages/` checkpoints raw original, verification, Rebuild and Review outputs. `records/` contains readable consolidated records and labels. `hidden/` stores only the 12 selected 1,536-dimensional vectors per example, not all activations or logits. `pilot_debug_v1/` preserves the initial boundary/truncation debug artifacts and is excluded from analyses. `selection_frozen.json`, `trained.joblib`, and `validation_grid.json` retain fitted settings. `test_predictions.json` retains paired predictions, labels and IDs. `test_results.json` reports AUROC, AP, Brier score, error recall at top 20%, false alarms, final answer accuracy, repairs and harm, and paired bootstrap intervals. Bootstrap intervals are conditional on this trained model and sample, not uncertainty over retraining seeds.

One GPU inference process runs at a time. Repair batches contain independent prompts; batch wall time is apportioned across examples and is throughput cost, not single-request latency. Offline data collection tries all repairs; deployment runs only the selected action. Cached original inference time is not counted as newly incurred cost. Stored group wall time includes scoring and extraction overhead; individual stage timing is reported separately. A and B detector inference runs on CPU.

## Apply saved models without references

`predict.py INPUT.json OUTPUT.json --hidden-dir PATH` accepts precomputed public feature records and per-ID hidden NPZ files; reference answers and observed action outcomes are unnecessary. See its module docstring for the input schema. `benchmark_apply.py` separately measures batch-one selector application on six excluded pilot questions.

`prepare_run.py` applies the final pilot-validated `experiments/ab/config.json` for a fresh run. The lower-level historical `pipeline.py prepare` retains the initial pilot defaults and is not the final-protocol entrypoint.
