# Official Train Feature Pool

## Scope

Extract all 7,473 official GSM8K main/train problems. Do not load or generate
the official 1,319-problem test split. Reuse verified original token IDs and
responses for 128 pilot, 1,000 exploration, and 1,000 confirmation examples.
Generate the remaining 5,345 responses once with the original FP32 Qwen2.5-1.5B
Instruct, fixed revisions, greedy decoding, original prompt and stopping rule.
Do not train Qwen or fit any new probe in this stage.

## Features

- Prediction checkpoints t=0,32,64,128, available only when generated length >t.
- Raw FP32 block outputs h7,h10,h13,h14 at these checkpoints, before final RMSNorm.
  These support h14-only, h14 plus adjacent difference, and multi-depth probes.
- Per-block residual norms, update norms, relative updates and cosine changes for
  all 28 blocks at the checkpoints. Only blocks <=14 are early probe inputs;
  later blocks are reserved for trajectory diagnostics.
- For blocks 7,10,14, entropy, probability margin p1-p2 and top1 probability at
  each prediction position 0..128 (or until generation ends). These retain all
  previously explored stability means/std/slopes and initial-to-current changes.
- Full generated responses and original prompt/generated token IDs, strict
  correctness labels, parse status, truncation and source provenance.
- No vocabulary-wide logits, full-token residual tensors, final-layer readout
  predictors, or full-response derived predictors are saved.

The model still generates each past token using all 28 blocks. Intermediate
readouts use the frozen final RMSNorm and LM head. Their computational overhead
is not a demonstrated early-exit speedup. t=0 is before the first response token.

## Quality and Storage

Independent full-prefix replay at available checkpoint positions must reproduce
at least 99% of greedy next-token argmax decisions in the first 32 examples.
Truncated-prefix h14 agreement checks guard against future-token leakage.
Imported generations must match the pinned dataset question/gold, exact chat
template prompt token IDs, decoded response and source hashes.

Write 32-problem ZIP batches, each with features.npz, generations.jsonl and a
small summary. Arrays contain flattened checkpoint rows and readout offsets.
The raw-vector ceiling is about 735MB uncompressed; total feature data is
expected around 1GB, not hundreds of GB. No per-token filesystem files are used.
At most one 32-example batch has per-example pending files.

Use /tmp/gsm8k-train-all-v1. Reuse the existing environment/cache under
/tmp/gsm8k-confirmation-v1 rather than installing another model/runtime.
Inference is a detached process, independent of SSH. A local background collector
receives every batch, verifies archive and member SHA-256 hashes, and only then
acknowledges deletion of that remote ZIP. Receipts remain for resumability.
If the collector disconnects, remote extraction continues and retains archives.
Stop safely with all committed and pending files preserved below 6GiB free disk.

## Analysis Boundary

This is an expanded training pool, not a new independent evaluation.
It includes previously used fitting, tuning and confirmation examples.
Before fitting the next probe, freeze and record a question-level split.
Do not present training-pool AUROC as generalization performance, or reuse the
previous confirmation AUROC after training on its examples.
The official test has already been inspected in earlier project experiments,
so it is not a pristine final holdout either. Record this limitation explicitly.
