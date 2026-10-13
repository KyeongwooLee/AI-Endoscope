# Published artifacts and interpretation

> Publication status: code, reports, figures and summary results are ready for upload. Raw response/feature/model bundles are prepared locally but are NOT uploaded; public export requires additional authorization after automatic approval review blocked it. Data restoration instructions below apply once those bundles are published.

Run `python restore_data.py` from this directory. The script combines the archive parts, checks SHA-256 for each part, archive and file, and restores `project/runs/`. It refuses to overwrite a changed file. `--verify-only` validates without extraction; `--output PATH` restores into another project root. Python 3.10+ standard library is sufficient.

The bundles contain original response text/token IDs, independent action outputs, verification outputs, confidence features, small hidden arrays, manifests, labels, fitted small classifiers, predictions, settings, source snapshots and diagnostic/audit history. They do not contain Qwen weights or full vocabulary logits. Raw records and trained model artifacts are compressed rather than individually listed in GitHub's browser. Reports and small summary files are also directly browseable.

## Runs

| Directory | Purpose |
|---|---|
| `20261009T0535Z` | Earlier debug/provenance run; not a completed A/B result |
| `20261009T0542Z` | Earlier GSM8K study supplying original-run historical inputs |
| `20261012_AB` | Original A/B: 40 pilot, 200 train, 80 validation, 100 official test |
| `20261012_followup` | Follow-up: 40 pilot, 1,200 training pool, 200 development, 300 fresh official test |

The follow-up excludes all locally previously sampled question hashes. Its final official-test subset excludes the original 100. These test outcomes are now published and inspected: do not treat them as untouched holdouts in another study. Train subsets are nested within each sample seed and overlap across the two seeds. No claim of independent repeated test sets is made.

## Labels and prediction inputs

- A: `label_error = 1` means the original answer is numerically incorrect.
- B: action order is Keep, Recalculate, Rebuild, Review. Four independent binary successes; unavailable Recalculate is `null` in records and −1 in evaluation arrays. Keep success = 1 − A error.
- Recalculate evaluates only the model's own safely extracted numeric expression. Its conservative extraction was not extended.
- Rebuild sees the original question; Review sees the question and original response. Actions are independent.
- Reference solutions, gold answers, repair outcomes and original correctness are excluded by the prediction-input whitelist. Recovery/damage diagnostics use correctness only to define groups, never as deployed input.
- Follow-up `hidden/*.npz`: `numeric_L14`, `boundary_L14`, `preverdict_L14`, each 1,536 coordinates. These are raw block14 states in verification context, before the verdict. Numeric position is reference-free; missing spans use zeros plus a shared availability feature.
- The scalar controls include response confidence/length, verifier judgment/NLL, numerical-answer confidence, both full CORRECT/INCORRECT label scores, missingness/truncation and calculator availability. Text is training-fitted TF-IDF. See [feature definitions](project/runs/20261012_followup/FEATURES.md).
- Primary follow-up scoring is v3. Legacy and strict-format scores remain separate sensitivities. V2 intermediate records are archived; they must not be mixed with final training records.

## Integrity and exclusions

`publication_manifest.json` lists every included archive member and SHA-256. Code and included research files preserve their local bytes. Generated Python caches, process-ID files, and separate source Notion/team-reference exports are excluded. Historical `artifact_sha256.json` and `original_preservation.json` retain their original meaning and may mention those omitted source documents; they are not rewritten to hide exclusions. Use the publication manifest for the downloaded package. Original run audits were completed locally before upload.

`.joblib` files are fitted small-model/scaler artifacts, not Qwen weights. Load them only from a trusted, checksum-verified source using the documented versions.

## Attribution

Questions/reference answers derive from [OpenAI GSM8K](https://github.com/openai/grade-school-math), dataset revision `740312add88f781978c0658806c59bc2815b9866`, under its MIT license (included in `GSM8K_LICENSE`). Responses/features were generated with [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Numerical scoring does not certify the reasoning or eliminate reference/extraction errors.
