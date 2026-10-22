# SmolLM2-1.7B detailed feature extraction v1

## Scope

This is a NEW generation run, not a replacement or reuse of Qwen responses.
No Qwen weights or verifier are fitted or applied here.

- GSM8K: all 7,473 official train questions, one greedy response each, same question
  text/system/user prompt as the prior Qwen full-train extraction. No official
  GSM8K test questions are loaded. The earlier "8K" pool was exactly 7,473.
- SVAMP: all original 1,000 questions, one sampled response each, T=0.7,
  top_p=0.95/top_k=0, matching the prior SVAMP sampling protocol.
- First 32 questions from GSM, then first 32 SVAMP, then remaining GSM/SVAMP.
  Original question order is preserved within each dataset. No outcome selection.
- Pinned SmolLM2-1.7B-Instruct revision resolved during prepare; FP32, SDPA,
  TF32 off, batch 1, max new tokens 1,024, original final-line/EOS stopping.
  Official model-specific chat template is recorded by SHA256.
- Every response is generated anew. Sampling seeds are deterministic ID hashes.
  Gold/reference is used only for labels, never supplied to generation/features.

## Detailed Storage

All saved floating-point vectors/features are FP32 unless noted. No quantization.
Raw hidden states mean block outputs BEFORE final RMSNorm; the native LM head
gets exactly one final RMSNorm. Intermediate readouts are diagnostics, not an
early-exit model or demonstrated causal correctness detector.

| Array | Shape for one response | Definition |
|---|---|---|
| hidden | C x 25 x 2048 | Embedding plus ALL 24 blocks at prediction t=0,32,64,128, when t < generated length |
| trajectory | T x 24 x 5 | Every generated prediction position, every block: input/output L2 norms, update norm, relative update, cosine distance |
| pooled_hidden | 24 x 3 x 2048 | Every block's end / whole mean / final-number mean completion representations |
| checkpoint_readouts | C x 24 x 7 | Per block: chosen-token logprob, entropy, p1-p2 margin, top1 probability, chosen-token rank, JS to final block, JS to previous block (embedding for block1) |
| checkpoint_top_ids/logprob | C x 24 x 10 | Top10 token IDs and log probabilities at every block/checkpoint |
| prefix_readouts | min(T,129) x 6 x 5 | Layers6/9/11/12/18/24, positions0..128: the five token distribution metrics above |
| token_readouts | T x 5 | Native final-block distribution metrics for the ENTIRE completion |
| token_top_ids/logprob | T x 10 | Native top10 IDs/log probabilities for every generated token |
| token_region | T | excluded/reasoning/final_number/format/unparsed codes |
| token_offsets | T x 2 | Exact Python Unicode character [start,end) offsets in decoded response; specials=-1 |
| token_masks | T x 3 | Non-special, whole nonwhitespace, final-number nonwhitespace masks |
| confidence | 26 | Prior whole/number 13-stat summaries of logprob, entropy, margin; empty masks remain NaN |
| length | 4 | log1p(prompt tokens, generated tokens, whole tokens, number tokens) |

T includes all generated token IDs, including a terminating special token when
present. t=0 is BEFORE the first response token; t consumes exactly t response
tokens. No future token is permitted to influence prediction-position features.
Canonical prompt-only forward states replace t=0 at ALL layers. Independent
short-prefix forwards check the middle layer12 at available checkpoints.

End is the last non-special completion token, possibly whitespace/newline. Whole
mean includes non-special nonwhitespace tokens, including final marker/number.
Number mean selects exact final-number overlapping token spans, or zero if empty.
Per-token native/readout probabilities are raw model probabilities BEFORE
sampling temperature/top-p changes; entropy uses natural logs, rank is 1-based
with strictly-greater logits determining ties. JS is in nats.

All raw vocabulary logits and full-token dense hidden tensors are NOT saved.
Checkpoint vectors support arbitrary later scalar features at those positions;
pooled vectors support probes at every depth, not new per-token vector pooling
definitions. Do not claim that every imaginable feature can be recovered.

## Quality And Boundaries

- Exact decode/token alignment, frozen source/input/model revisions, FP32 finite
  feature checks, final-native projection and generation logprob comparisons.
- First 32 GSM examples must have >=99% greedy checkpoint argmax agreement
  plus independent truncated-prefix checks before continuing. Rechecked on resume.
- Sampling examples are NOT subjected to greedy argmax agreement.
- Final numerical Decimal comparison uses the original strict final-line parser;
  format failures and truncation remain separate. This is not human review of
  reasoning or reference validity. No error-type labels are inferred here.
- Future-token invariance and array/mask shapes are unit-tested on a small random
  Llama model. Real GPU QA is still required and recorded by the running worker.
- Smol layers map by relative depth, not literal Qwen layer number. Hidden space
  differs: future probes must be trained separately, not reused Qwen weights.
- This is a public, previously studied train pool, not a clean new benchmark.
  Freeze question-group splits before fitting; repeated questions cannot cross folds.

## Disk, Resume, Backup

Use /tmp/smollm2-detailed-v1. Reuse system GPU PyTorch via a separate venv; all new
model caches, runtime packages and data are under /tmp. No root cache growth is
intended. Guard at 35GiB total study, 18GiB unbacked outbox, and 6GiB free space.
These are below the user-provided 50GB limit even if the host reports more space.
Expected output is roughly 8-15GB, not hundreds of GB; actual batch size is recorded.

Sixteen responses per batch ZIP, each with per-response compressed NPZ plus
generations.jsonl. NPZ includes a JSON record with question/gold, generation token
IDs/text/config/seed, label, confidence names and QA. No per-token filesystem files.
Keep completed generation JSON before replay, so extraction interruption does not
require resampling. Per-example pending files are removed after committed batch
archive AND member hashes pass local remote-side verification.

Local collector verifies the downloaded ZIP AND every member SHA256 before ACK.
Only that named remote ZIP is then deleted; receipts/ACKs remain for resume.
If SSH/collector disconnects, detached inference continues and retains unbacked
ZIPs until its disk cap. Disk-guard failures preserve all data for collect/resume.
Do not delete remote caches, pending files or receipts to force continuation.

## Files And Running

- prepare.py freezes model revision and verifies original local question sources,
  then makes input-bundle.zip and READY.json under outputs/...-prepared.
- deploy.py verifies bundle, rejects duplicate worker, launches detached process.
- run.py owns the lock, generation, QA, feature extraction, commit and resume.
- extract.py accepts NO gold or correctness input.
- control.py exposes inventory, metadata snapshot and exact verified ACK deletion.
- collect.py reconnects and backs up while the local machine/app can run it.
- bootstrap.sh creates /tmp runtime using existing GPU PyTorch.

```text
python prepare.py --workspace <original-local-workspace>
python -m unittest -v test_details
# Copy input-bundle.zip, READY.json, deploy.py to the named remote root.
python deploy.py
python collect.py --host root@HOST --port PORT --known-hosts PATH --local BACKUP
```

README/config are design; actual progress, quality_gate.json, environment.json,
RUN_MANIFEST.json and verified local receipts are evidence of execution. Never
infer completion just because deployment returned a PID. Model inference is not
training, and this stage does not fit or validate any new classifier.

Original GSM8K/OpenAI and SVAMP/Arkil Patel MIT notices are bundled in src/.
Model source: https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct .
Copied engine/answer/common helpers are frozen snapshots of the existing project
code; prepare's source hashes identify their exact versions.
