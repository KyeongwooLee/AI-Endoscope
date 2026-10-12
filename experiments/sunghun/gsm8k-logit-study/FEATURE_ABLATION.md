# Depth-14 Feature Ablation

Compare engineered features without inspecting or evaluating the old test again. Reuse
the existing raw-vector extraction and immutable per-token entropy/margin readouts.

## Inputs and Cohort

All comparisons use the same exploration problems still active after 32 generated tokens:
799 original train problems and 200 original validation problems. Inputs stop at decoder
block 14. Full-depth scalar trajectory files are immediately sliced to blocks 1..14.
No generated-token support/rank, final-layer confidence, answer region, total response
length, gold number, or correctness-derived predictor is used.

Engineered groups:

- Size (10): raw h14 norm, mean/std/RMS, absolute maximum and quantiles, coordinate
  energy concentration and effective participation fraction.
- Adjacent (6): h13-to-h14 absolute/relative/directional update, norm ratio, update
  alignment with previous/current residual.
- Flow (13): h7-to-h10, h10-to-h14 and h7-to-h14 changes, norm ratios, and alignment
  between the two update vectors. Unequal depth spans are acknowledged explicitly.
- Acceleration (18): slopes, local differences, ratios, decreasing-step fractions and
  average second differences of scalar update magnitudes/directions through block 14.
  These are NOT second differences of full hidden vectors; h12 is not stored.
- Stability (42): current/prefix/last-eight-token mean, std, slope, and change from
  the first prediction for entropy and margin at layers 7, 10 and 14.

Compare each group, all current structural groups, all current groups, initial-only
engineered features, initial plus current, and initial plus changes. Raw h14 initial,
current, and concatenated initial/current probes are reference baselines.

Initial and current representations belong to DIFFERENT prediction positions and contexts.
Their difference is not a change to the same token, and cannot identify the onset of a
reasoning mistake by itself. Initial-plus-changes is a representation choice, not an
independent information source over initial-plus-current.

## Training and Comparison

Split the 799 train problems into 639 fit / 160 tune, stratified with seed 2026. Fit
coordinate standardization on fit only. Select hyperparameters and early-stop epochs
using the 160 inner-tune problems only. Then refit standardization and models on all 799
train problems for the preselected number of epochs. No outer-validation early stopping,
hyperparameter selection, feature selection, or threshold fitting occurs.

Train linear and 32-unit MLP probes on the RTX 3090, with seeds 42/43/44. Linear candidates
use learning rate .003 and AdamW decay .1/1; MLP candidates use rate .0003/.001 and decay
.1. Both use weighted BCE, with at most 300 inner-fit epochs and patience 30. Score each
ensemble on the original 200 validation problems only after all models are frozen.

Report AUROC, AP, pointwise problem-bootstrap intervals, and precision/recall for the
predefined highest-risk 20% of problems. Compare initial-only to initial-plus-current or
changes with paired bootstrap. Record every candidate setting and individual seed AUROC.

Univariate direction is determined on train only. Export train/validation univariate
diagnostics and train Spearman correlations to expose redundancy. Do not interpret the
best validation feature among many scans as a confirmed discovery.

## Limitations

The original validation was already seen in previous experiments. This comparison is
exploratory despite nested tuning, and multiple comparisons are not corrected. Any feature
or group selected from these results requires a fresh held-out evaluation. Scores are not
calibrated probabilities, and no answer-accuracy intervention is performed.

Use `scripts/prepare_feature_ablation.py` locally to create the compact exploration-only
input, then `scripts/train_feature_ablation.py` in the existing VESSL GPU environment. Run
detached, back up the full result with SHA256 verification, and remove only verified remote
data artifacts. Further residual extraction is not required for these feature groups.
