# Source audit — grounded error-description pilot (pilot_v1)

Machine-readable versions: `runs/pilot_v1/asset_manifest.json` (asset paths, bytes, SHA256, branch heads observed at run time), `runs/pilot_v1/audit.json`, `runs/pilot_v1/overlap_audit.json`, `runs/pilot_v1/prior_question_exclusions.json`. The `runs/` and `data/` directories are local only (git-ignored).

## Repository and branch

- Local clone: `/home/coder/yaicon-errorprobe/AI-Endoscope`, new branch `minseon-error-description-pilot` created from `origin/minseon` @ `73d02635b55358c2b942c2bcd084ebb334c0db85`. No other branch was merged or modified; nothing was pushed.
- The user's working directory `/home/coder/yaicon-errorprobe` is a non-git copy of Minseon's earlier A/B and follow-up runs. It was read (question IDs only, for exclusion checks) and not modified, apart from the new local virtual environment `/home/coder/yaicon-errorprobe/.venv-edp`.
- No `AGENTS.md` exists in the clone.

## Branch pins (checked with `git ls-remote` at audit time)

| Branch | Pinned in handoff | Observed | Changed |
| --- | --- | --- | --- |
| main | `51560936fa89329e481a2c8c8f7af223df27576a` | same | no |
| dahyeon | `51560936fa89329e481a2c8c8f7af223df27576a` | same | no |
| minseon | `73d02635b55358c2b942c2bcd084ebb334c0db85` | same | no |
| minsu | `d95116f0d8a6b64a7f3eb08457af11144d6f4a7c` | same | no |
| sunghun | `99b35512ba043b0ec86d809ec700fcb4ba969889` | same | no |
| kyeongwoo | `7c4485dbc3f14e3771210fe626efe155e8eb468e` | same | no |

No adopted revision differs from the handoff baseline.

## Assets (all SHA256-verified before parsing)

| Release / source | File | Bytes | SHA256 |
| --- | --- | ---: | --- |
| qwen-gsm8k-self-errors-v1 | errors.jsonl | 12,942,210 | `cc54b199…2c1993b` |
| qwen-gsm8k-self-errors-v1 | features_meta.json | 122,120 | `b0d6ba83…be60464` |
| qwen-gsm8k-self-errors-v1 | DATA_SCHEMA.json, README.md, SHA256SUMS.txt, ATTRIBUTION.md, LICENSE-GSM8K.txt | small | as pinned in `config.json` |
| error-taxonomy-full-v1-draft | labels.jsonl | 2,938,907 | `6137ffc9…c23193d` |
| minsu @ d95116f (raw.githubusercontent) | taxonomy_relabel/setup_ab/out/ab_labels.tsv | 64,867 | `dfa22bef…fdd64ae` |
| minsu @ d95116f (raw.githubusercontent) | taxonomy_relabel/loc_audit/out/loc_labels.tsv | 47,319 | `6f1529c4…db7dd77` |

Full hashes are in `config.json` (`sources.*.assets`) and `asset_manifest.json`. Not downloaded (not needed for the text pilot): `hidden.npy`, `token_ids.jsonl`, `confidence.npy`, Qwen weights, SAE weights. The embedding model `BAAI/bge-m3` @ `5617a9f61b028005a4858fdac845db406aefb181` (weights `pytorch_model.bin`, tokenizer, configs; no ONNX/ColBERT/sparse heads) was fetched into the user's Hugging Face cache.

## Recomputed audit (matches the handoff table exactly; 0 mismatches)

| Property | Value |
| --- | ---: |
| Responses / unique IDs | 7,691 / 7,691 |
| Unique questions | 3,489 |
| Sampled / greedy | 6,380 / 1,311 |
| finish=stop / finish=length | 7,656 / 35 |
| Extra copies of identical (pid, response) | 5 |
| Identical response text across pids | 0 |
| Normalized question groups spanning pids | 0 |
| features_meta ID order = row order | yes |
| Rule counts | SETUP 4,422; NOEQ 1,797; TARGET 586; OTHER 549; ARITH 311; NEWNUM 26 |
| finish=length rows that still contain `####` | 35 (all; excluded anyway) |

Additional checks: every row has `id == train-{pid}-{sample}`, `split == train`, `mode` consistent with `sample`, finite `gold`/`pred`, and `|gold − pred| ≥ 1e-6` (all rows are wrong under the release rule). Within each pid, question/reference/gold are identical. JSONL is parsed by LF-delimited byte records (U+2028-safe).

## Shared SAE split

`GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)` on the full ordered 7,691 rows with `pid` groups (Kyeongwoo `prepare.assign_splits`), scikit-learn 1.7.2 (the version pinned in `sae_error_clustering/requirements.txt`): discovery 6,114 / 2,791, confirmation 1,577 / 698 — identical to the handoff. The complete map is `runs/pilot_v1/sae_split_map.json` (same key/value format as Kyeongwoo's `splits.json`). Cross-partition normalized-question conflicts: 0.

## Prior annotation / explanation overlap

| Source | What it is | Overlap with primary release | Used as exclusion |
| --- | --- | --- | --- |
| Sunghun `error-taxonomy-full-v1-draft/labels.jsonl` | 1,100 AI-draft annotations (896 GSM8K, 204 SVAMP), separate responses | 201 questions / 467 rows; 1 exact normalized-question + unchanged-response pair | yes (201 question groups) |
| Minsu `taxonomy_relabel/setup_ab/out/ab_labels.tsv` | AI-review rubric reasons on this release's `train-{pid}-{sample}` IDs | 814 rows / 814 questions (all IDs matched) | yes |
| Minsu `taxonomy_relabel/loc_audit/out/loc_labels.tsv` | AI first-error location reasons on this release | 513 rows / 513 questions (all IDs matched) | yes |
| Local `runs/20261012_followup/manual_adjudications.json`, `manual_audit_sample.json`, `PILOT_EXAMPLES.md` (and other local reports scanned) | Minseon's manually reviewed / displayed train examples (different responses) | 17 + 17 + 4 question matches verified by question text | yes (conservative) |
| Local `runs/20261012_followup/records`, `runs/20261012_AB/records` | Minseon's own Qwen generations (not annotations) | 846 overlapping questions | no (informational) |

Total excluded question groups: 1,306. With only length + `labels.jsonl` exclusions the eligible counts are 2,612 discovery / 663 confirmation questions (reproduces the handoff); with all exclusions 1,752 / 418. Minsu's `relabeled.tsv` / `sh_audit` files use Sunghun-format IDs (`train-00480__T070__a04`) from the 1,100-annotation set and are covered by the `labels.jsonl` question exclusion.

Not checked: team-discussion explanation attempts that are not present as runnable artifacts in the inspected branches; Dahyeon's annotation work (branch tip equals main); Sunghun's other releases (different responses, not the corpus). Absence of public artifacts is not taken as proof that such work did not happen.

## Known limitations of sources

- Rule labels (`rule_type`) are automatic and not human-reviewed; `NOEQ` means no checkable equation was parsed, not absent reasoning. They were used only to enrich the 10-item development stress sample and never shown to annotators or reviewers.
- GSM8K references are fallible; the annotations flag 8 evaluation items (E and G each) where the question wording or reference is questionable.
- The legacy annotations have no exact judge model/version record; none of their text was used.
