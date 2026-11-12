# Joining this pilot to Kyeongwoo's SAE analysis

`runs/pilot_v1/sae_bridge.jsonl` has one row per pilot response (30 development + 100 evaluation):

```text
source_release, source_asset_sha256, source_row_index, source_id, source_pid, question_key,
response_sha256 (exact UTF-8, unnormalized), canonical_response_id, all_source_ids,
sae_partition, pilot_role, E_status, G_status, human_review_status, sae_features_attached=false
```

No SAE vectors or clusters are attached; none were computed or invented. `runs/pilot_v1/sae_split_map.json` holds all 7,691 `source_id → discovery|confirmation` assignments in the same format as `sae_error_clustering`'s `splits.json`.

## Contracts in `sae_error_clustering` @ `7c4485db` (inspected, not executed)

- `prepare` → `splits.json` (id → partition), `prepared.jsonl` (id, pid, response, token IDs, partition, canonical_id).
- `extract` → `shards/*.npz` with `ids`, `sae_mean` (n × 24,576), `hidden_mean` (n × 1,536), `peaks`; each shard has a JSON with signature, SHA256 and per-row quality. `extraction.json` has `complete`.
- Hook: zero-based `model.layers[17]` output (18th block), SAE applied per token then averaged. Not HF `output_hidden_states[17]`; the release `hidden.npy` cannot substitute. Release h28 is post-final-RMSNorm; Sunghun's h28 is pre-norm — never pool them.
- `cluster` → `clusters.npz` (`used_indices`, `discovery_indices`) indexes into its own row order; use `report/assignments.csv` (ID-addressable) instead of positional indices.

## Join procedure (when a completed run exists)

1. Set `"sae": {"results_dir": "/absolute/path/to/sae_error_clustering/results"}` in a copy of `config.json` with a new `run_dir`, or call `edp.cluster_bridge.check_sae_run` directly.
2. `run.py --stage bridge` validates: `splits.json` equals our map for every ID (raises otherwise), `extraction.json.complete == true` (otherwise reported `partial_run_not_used`), and counts pilot IDs present in `assignments.csv`.
3. Join features only where `source_id` matches **and** `sha256(prepared.response) == response_sha256`. Map duplicate rows through `canonical_response_id`. Report every coverage loss (extraction exclusions, truncation, alignment failures).
4. Do not use E/G descriptions or their clusters as ground truth for SAE. A declared comparison on the same targets: rank the same 80 gallery responses for the same 20 queries by cosine of `sae_mean` (and `hidden_mean`) without refitting, then judge any new query–neighbor pairs with the same blinded pair protocol; compare verified_same_error@5 to R/E/G on the shared reviewed pairs.
5. If only partial shards exist, label every number as partial; do not bypass the loader's completeness check to report full-run results.
