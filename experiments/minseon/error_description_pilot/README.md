# Error-description embedding experiments (Minseon)

## Current GPT experiment

The current report covers GPT-6-astra annotations of 630 distinct GSM8K problems with stored Qwen2.5-1.5B-Instruct solutions: a 130-case pilot and a disjoint 500-case expansion. Each solution receives ordinary and evidence-grounded annotations; eligible descriptions are embedded with BGE-M3 and analyzed with nearest-neighbor retrieval and clustering.

- [Korean report with examples and cluster browser](runs/gpt6_rerun_v1/analysis_report.html) — download the run directory and open this HTML file in a browser, keeping its images alongside it.
- [Complete result archive](runs/gpt6_rerun_v1_results.zip)
- [Run instructions and dependencies](runs/gpt6_rerun_v1/RUN_INSTRUCTIONS.txt)
- [Validation results](runs/gpt6_rerun_v1/validation.json)
- [Exact GPT requests and responses](runs/gpt6_rerun_v1/calls/)
- [Experiment scripts](runs/gpt6_rerun_v1/source/)

All 1,260 annotations, embeddings, 3,674 unique pair judgments, and 500 reference-free diagnoses are saved. Judgments are provisional AI assessments, not independent human ground truth. Annotation and evaluation use the same requested GPT model in separate contexts. The exact provider model snapshot was not exposed.

The report presents GPT results only. Earlier run artifacts remain available in `runs/pilot_v1/`, `runs/expansion_500_v1/`, and `runs/expansion_500_v2/` for provenance and reproducibility. Source data and run artifacts are intentionally committed; credentials, model weights, virtual environments, and Python bytecode are not included.

## Original pilot protocol and implementation

Do short descriptions grounded in the actual solution (G) retrieve wrong answers with the **same observable mathematical mistake** better than ordinary short error explanations (E), the raw question+solution text (R), the question alone (Q), or random? Existing Qwen2.5-1.5B-Instruct GSM8K wrong answers (`qwen-gsm8k-self-errors-v1`), BGE-M3 dense embeddings, a fixed 20-query × 80-gallery evaluation, and blinded human pair review.

**Status (pilot_v1): all automated stages complete; human review pending; semantic metrics are `null`.** E and G were written by the same agent session (`condition_context_isolated=false`), so any future G−E result from this run is exploratory, not the protocol-compliant primary contrast.

- Results (Korean): [runs/pilot_v1/RESULTS_KO.md](runs/pilot_v1/RESULTS_KO.md)
- Reproduce: [REPRODUCE.md](REPRODUCE.md) · Sources: [SOURCE_AUDIT.md](SOURCE_AUDIT.md) · Reviewers: [REVIEWER_GUIDE.md](REVIEWER_GUIDE.md) · SAE: [SAE_JOIN.md](SAE_JOIN.md)
- Prompts: [prompts/E_v1.txt](prompts/E_v1.txt), [prompts/G_v1.txt](prompts/G_v1.txt), [history](prompts/PROMPT_HISTORY.md)

## Layout

```text
run.py              CLI (stages: test, audit, sample, export-packets, import-annotations, freeze,
                    embed, retrieve, review-package, cluster-dev, bridge, report, all-auto, evaluate-reviewed)
config.json         sources + SHA256 pins, split, sampling, annotator, embedding, retrieval, review, bootstrap
edp/io.py           hashing, LF-safe JSONL, atomic/guarded writes
edp/audit.py        download + checksum, schema validation, audit counts, prior-annotation overlap
edp/sampling.py     shared SAE split (GroupShuffleSplit seed 42), eligibility ledger, 30 + 100 sample
edp/annotation.py   packet whitelist, prompt rendering, import/validation, exact span resolution
edp/embedding.py    BGE-M3 dense, token-length check (no truncation), hash-keyed cache
edp/retrieval.py    cosine top-k, tie-break by source_id, abstention handling, seeded Random
edp/review.py       blinded item/pair packages, private key, spreadsheet-safe CSV, edit preservation
edp/evaluate.py     judgment resolution, verified_same_error@5, paired bootstrap, refusal on missing
edp/cluster_bridge.py  development K-means diagnostic, sae_bridge.jsonl, SAE run validation
edp/report.py       status.json and RESULTS_KO.md from actual outputs
tests/test_pilot.py 23 unit tests on synthetic fixtures
data/, runs/        versioned source assets and experiment outputs
```

Scope limits: reference-assisted offline analysis of numeric wrong answers on already-studied GSM8K train data; no detection AUROC, no causal claim, no repair claim. Consult each run's report for its own status and limitations.
