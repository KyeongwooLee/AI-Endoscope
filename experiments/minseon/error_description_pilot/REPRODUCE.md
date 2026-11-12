# Reproduce the grounded error-description pilot

All commands run from the repository root (`AI-Endoscope/`). Paths in `config.json` are relative to this package directory. Nothing here calls a paid API, regenerates Qwen answers, or needs the SAE/hidden-state archives.

## 1. Environment

```bash
# Python 3.10+; this run used a venv on top of the system torch 2.12.1+cu130.
# (python3-venv's ensurepip was unavailable, hence --without-pip + the system pip module.)
python3 -m venv --without-pip --system-site-packages ../.venv-edp
../.venv-edp/bin/python -m pip install -r experiments/minseon/error_description_pilot/requirements.txt
export PY="../.venv-edp/bin/python -I"   # -I: isolated mode; downloaded data never shadows modules
```

`run.py` sets `USE_TF=0` before importing `transformers` (a system TensorFlow/Keras 3 install otherwise breaks the import).

## 2. Tests (CPU, synthetic fixtures only)

```bash
$PY -m unittest discover -s experiments/minseon/error_description_pilot/tests -v
# or, recording the result into the run directory:
$PY experiments/minseon/error_description_pilot/run.py --stage test
```

23 tests: checksum/ID/order mismatches, duplicate IDs, exact-duplicate mapping, SAE split reproduction, sample disjointness and exclusions, LF-safe Unicode JSONL, exact response hashes, character spans with repeated quotes and non-ASCII text, packet whitelists, single-pass prompt rendering, embedding-input isolation, abstention, empty galleries, ties, vector shape/NaN/normalization checks, pair blinding/dedup, shared judgments, unreviewed vs uncertain, AI-review exclusion, unresolved disagreements, hand-checkable bootstrap, attempt limits, frozen-prompt refusal, and guarded resume writes. Synthetic fixtures do not establish real-data accuracy.

## 3. Automated pipeline

```bash
$PY experiments/minseon/error_description_pilot/run.py --stage all-auto
```

`all-auto` runs: test → audit (download if missing + SHA256 verify) → sample → export/import development packets → freeze check → export/import evaluation packets → embed → retrieve → review-package → cluster-dev → bridge → report. Each step that lacks a prerequisite is recorded as pending in `runs/pilot_v1/all_auto_steps.json` and the run continues with independent steps. It never fabricates annotations or reviews.

Individual stages (same order):

```bash
R="$PY experiments/minseon/error_description_pilot/run.py --stage"
$R audit
$R sample
$R export-packets --phase dev
#   annotator writes runs/pilot_v1/annotation/inbox/dev/{E,G}/attempt1.jsonl (optional attempt2.jsonl)
$R import-annotations --phase dev
$R freeze                       # freezes prompts, annotator, embedding, retrieval, metrics, manifests
$R export-packets --phase eval  # refuses if prompts differ from protocol_frozen.json
#   annotator writes runs/pilot_v1/annotation/inbox/eval/{E,G}/attempt1.jsonl
$R import-annotations --phase eval
$R embed
$R retrieve
$R review-package
$R cluster-dev
$R bridge
$R report
```

### Annotation contract (file-based; no helper calls any model)

- Packets: `runs/pilot_v1/annotation/packets/{phase}/{E,G}.jsonl` (+ readable `.md`). Inputs are whitelisted: source_id, response_sha256, question, unchanged response, reference. Rule labels, legacy annotations, `has_correct_sibling`, features, confidence and clusters are rejected.
- Outputs: one JSON object per line per packet, in `inbox/{phase}/{cond}/attempt1.jsonl`; at most one corrective `attempt2.jsonl`. G evidence may give `evidence_start/end` (Python char offsets, half-open) or `evidence_occurrence` (1-based); a repeated quote without either is rejected. Quotes are never normalized.
- Every attempt is appended to `annotation/attempts.jsonl`; accepted overlays go to `annotation/validated/{phase}/{cond}.jsonl` with a cache key over (response hash, condition, prompt hash, schema, model, route, settings).
- In pilot_v1 the executing agent (`claude-opus-5-5`) wrote E and G directly in one session: `condition_context_isolated=false`. For a protocol-compliant G−E, re-run the same packets for E and G in separate model contexts (e.g. two fresh sessions given only `E.md` or only `G.md`) into a **new** `run_dir`.

## 4. Human review (pending)

1. Read `REVIEWER_GUIDE.md` (also copied to `runs/pilot_v1/review_package/`).
2. Copy `runs/pilot_v1/review_package/templates/reviewer_1/` (and `reviewer_2/` for the second reviewer's prespecified subset) to `/absolute/path/to/completed_reviews/<reviewer_id>/`; fill only the judgment columns; set `reviewer.json` (`reviewer_type: "human"`).
3. Stage 1 (`items_stage1.csv`) before Stage 2 (`items_stage2.csv`); pairs in `pairs.csv`. Do not open `runs/pilot_v1/review_key_private/`.
4. Optional adjudication: `adjudication_template.csv` → `pair_code, final_judgment, adjudicator, notes`.

```bash
$PY experiments/minseon/error_description_pilot/run.py --stage evaluate-reviewed \
  --reviews /absolute/path/to/completed_reviews \
  [--adjudication /absolute/path/to/adjudication.csv]
```

Blank judgments are missing, not zero; any missing required pair keeps the primary metric `null` (`human_review_pending`). AI reviews must be marked `reviewer_type: "ai_provisional"` and are never counted. Re-running `review-package` never overwrites an edited file.

## 5. Determinism and resume

- Split: scikit-learn GroupShuffleSplit, seed 42. Sampling: `numpy.random.default_rng([20261108, stream])`, groups first, then one response per group, stable sorted IDs.
- `fingerprint.json` (sources, split, sampling) and `protocol_frozen.json` refuse silent changes; use a new `run_dir` for any protocol change.
- Embeddings are cached per (model revision, precision, exact text SHA256) under `runs/pilot_v1/embeddings/cache/`; a rerun re-encodes nothing.
- Measured on RTX 3090: full embedding stage 18.7 s including model load, 2.36 GB peak GPU, 3.32 GB peak RSS. CPU execution is supported (`"device": "cpu"`) but was not timed.
