# Raw Hidden-State Probes and Layer Changes

This is an exploratory extension of the frozen Qwen2.5-1.5B-Instruct FP32 GSM8K run.
No language-model weights are trained and no answers are regenerated.

## Protocol

- Fixed model revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- Replay saved prompt IDs and generated prefix IDs with FP32 SDPA, TF32 disabled.
- Primary checkpoint: 32 prior generated tokens, predicting token 33 at decoder block 14.
- Secondary checkpoints: 0, 64 and 128. Only still-active generations are available.
- Preserve original 800/200 exploration train/validation split and 1,319 test problems.
- Target: incorrect final answer, including parsing failures and truncations.
- Raw hidden vectors are captured BEFORE final RMSNorm. No selected next token is an input.

## Three Inputs

1. `h14`: 1,536 values.
2. `[h14, h14-h13]`: 3,072 values, with the immediately preceding update explicit.
3. `[h7, h10, h14]`: 4,608 values across three available depths.

The second representation has exactly the same information as `[h13,h14]`, but a different
parameterization. It does not reconstruct the entire preceding layer trajectory.

Train a linear probe and an MLP with a 32-unit hidden layer for each input. Fit coordinate
standardization on training problems only, then clip to +/-10. Use weighted binary loss,
AdamW and validation AUROC early stopping. Linear learning rate is .003 with decay
.01/.1/1; MLP learning rates .0003/.001 with decay .1/1. Seeds 42/43/44 are averaged for
selection and evaluation; save individual seed scores too. Final thresholds maximize
validation F1. Freeze all settings before test evaluation.

MLP parameter counts per seed: 49,217 / 98,369 / 147,521. Report the seed ensemble separately:
a deployment using one seed has one third of the ensemble's probe computation. GPU training
does not itself establish runtime overhead, calibration, or improved answer accuracy.

## Layer Changes

At each fixed token position, measure all raw block transitions, including embedding h0:

- Absolute update: `||h_l-h_(l-1)||`.
- Relative update: absolute update divided by `||h_(l-1)||`.
- Direction change: `1-cosine(h_l,h_(l-1))`.
- Residual vector norms, to distinguish smaller relative updates from growing residual scale.

Plot correct/incorrect problem means with problem-bootstrap pointwise confidence intervals.
Use layers 2..28 for descriptive constant, 1/x and power-law fits; layer 1 is a distinct
embedding transition. Fit curves on exploration means and evaluate the fixed fits against
test means. Do not assume a decreasing curve or exponent -1. Neither shrinking updates
nor a visually separated curve is sufficient to establish error classification performance.

## Storage and Recovery

Only four selected layer vectors at four token checkpoints are retained (approximately
228MB uncompressed across 2,319 problems), not all residuals. Full-depth trajectories are
scalar summaries. Batches contain 32 problems and have SHA256 receipts. Resume validates
configuration and completed batch hashes. First-batch native next-token argmax agreement
must exceed .99, and a truncated-prefix check verifies causal h14 replay.

Run in `/tmp/gsm8k-hidden-study` using a detached process, then copy each completed batch
locally and verify its hash. Keep remote batches until training/analysis finish because they
remain inputs to those stages. After the final local backup is verified, delete only the
experiment's remote data files. Do not touch unrelated `/tmp` content.

## Limitations

The test set has already been inspected in previous analyses, so all results are exploratory.
Confidence intervals condition on fitted models, are not corrected for multiple comparisons,
and do not establish robustness on new distributions. The experiment uses original full-model
prefixes, not trajectories changed by a live controller. A later controlled intervention and
an untouched evaluation set are necessary to demonstrate higher answer accuracy.
