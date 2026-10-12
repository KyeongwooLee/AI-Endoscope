# Frozen Entropy Candidates on Fresh Problems

## Objective

Check whether the three exploratory entropy features reproduce on problems not previously
used in this project. This is not a retry/intervention experiment and does not train Qwen.

## Fixed Choices Before Fresh Evaluation

- Model revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, FP32 SDPA, TF32 disabled.
- Dataset revision `740312add88f781978c0658806c59bc2815b9866`.
- Original prompt, greedy decoding, maximum 1,024 generated tokens and original final-line stop.
- Select 1,000 remaining GSM8K train problems, seed 20261009, excluding all earlier pilot,
  exploration and test IDs and normalized-question duplicates.
- Primary score: negative layer-10 entropy averaged over prediction positions 0..32.
- Secondary scores: negative layer-7 mean entropy; negative difference between layer-7
  mean entropy and initial entropy; a three-feature linear/MLP probe; frozen raw initial h14 MLP.
- Target: incorrect final answer including parse failures/truncations under the original parser.
- All primary comparisons use the same still-active t32 population. Report coverage and
  first-token baseline results without treating early-completion absence as a predictor.

## Fitting

Single-feature directions (-1) are frozen from previous exploration. Train the three-feature
combination on the original 799 training examples only. Use the old 639/160 train-internal
split for hyperparameters and epochs, then refit on 799 with fixed epochs. The original
validation is not used for new tuning. The raw initial MLP and its scaler are reused unchanged
from the prior feature ablation. All fitted artifacts are committed BEFORE fresh questions
are loaded. No test feedback changes features, direction, thresholds or models.

## Capture and Quality

Capture raw block outputs during the actual cached generation at layers 7/10, prediction
positions 0..32; capture the initial layer-14 raw vector. Apply the frozen final RMSNorm/LM
head to intermediate vectors to compute entropy. This uses static weights, not hidden
activations from layer >14. Next-token choice and gold are never feature inputs.

The first eight new problems additionally check cached-head argmax against actual greedy
tokens, independent full-prefix replay argmax agreement >=.99, entropy discrepancy <.001,
and raw initial h14 agreement within numerical tolerance. Failed checks stop the run and
remain failed on resume. Source/model/input hashes prevent mixing changed code with old data.

Records are committed atomically; resume verifies completed records and avoids regenerating
them. Every 32 problems are packaged with SHA256 receipts for local backup. Full residuals
and full logits are not retained. The expected server footprint is about 10GB including
the recreated GPU environment and model cache; experiment outputs are only tens of MB.

## Evaluation and Limits

Report AUROC and AP, problem-bootstrap intervals, and precision/recall at a predefined
top-20% risk budget. Primary inference concerns the predeclared layer-10 mean score.
Other feature comparisons are secondary and unadjusted for multiplicity. Report a null or
weaker result without changing the protocol. Compare against the frozen first-token h14 baseline.

These problems are new to this project, not necessarily absent from model pretraining.
Intermediate logit-lens entropy is not calibrated answer confidence. Risk discrimination
does not demonstrate higher answer accuracy, real-time overhead, or detected reasoning-error onset.

## Remote Run

Dedicated directory: `/tmp/gsm8k-confirmation-v1`, detached shell with stdin closed and logs
written to `job.log`. `progress.json` tracks generation/capture completion; `quality_gate.json`
confirms numerical validation. `outbox/` holds 32-problem archives. After 1,000 problems,
evaluation runs automatically and `final-backup.zip` is created with member hashes.

Keep remote records until the final archive is verified locally. Do not delete environment,
cache or records while inference is active. No SSH session is required to keep the run alive.
