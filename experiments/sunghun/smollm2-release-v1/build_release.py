"""Verified, lossless full shards and Qwen-equivalent compact feature packaging."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import runpy
import shutil
import zipfile

import numpy as np

from load_data import CHECKPOINTS, RAW_LAYERS, READOUT_LAYERS, SCALAR_NAMES, scalar_features, sha256

FINGERPRINT = "dc40716a9fffcb546b478c759ba5e9a90546c9481ec830c53051f9059c342a2c"
METADATA_SHA = "c71a036950c93aea79a3135f9d8fdaf1028dfb2f6ea8af7e17bc1ce908e9be2e"
RECOVERY_FINGERPRINT = "75d50117968c998f82ddb6aad71d55f3c56f8dfbc6037b3cfa25e497bf963603"
COMPACT_KEYS = {"ids", "checkpoints", "hidden", "trajectory", "readout_ids", "readout_offsets", "readouts"}
FULL_KEYS = {"record", "checkpoints", "hidden", "pooled_hidden", "trajectory", "checkpoint_readouts",
             "checkpoint_top_ids", "checkpoint_top_logprob", "prefix_readouts", "token_readouts",
             "token_top_ids", "token_top_logprob", "token_region", "token_offsets", "token_masks", "confidence", "length"}


def encode(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_snapshot(path, expected_sha=None):
    if expected_sha is not None:
        require(sha256(path) == expected_sha, "Immutable metadata hash mismatch")
    with zipfile.ZipFile(path) as z:
        contents = {name: z.read(name) for name in z.namelist()}
    manifest = json.loads(contents["snapshot_manifest.json"])
    require(set(contents) == set(manifest) | {"snapshot_manifest.json"}, "Unexpected metadata members")
    for name, expected in manifest.items():
        require(digest(contents[name]) == expected, "Metadata checksum: " + name)
    return contents


def recovery_inputs(source, original, receipts, recovery):
    """Select source batches explicitly; never substitute or relabel original bytes."""
    selected = [(r, source / "incoming" / (r["name"] + ".zip")) for r in receipts]
    if recovery is None:
        return selected, None
    contents = read_snapshot(recovery / "final-metadata.zip")
    manifest = json.loads(contents["RUN_MANIFEST.json"])
    require(manifest["fingerprint"] == RECOVERY_FINGERPRINT, "Unapproved recovery fingerprint")
    require(hashlib.sha256(json.dumps(manifest["files"], sort_keys=True).encode()).hexdigest() == RECOVERY_FINGERPRINT,
        "Recovery fingerprint does not match its frozen hash table")
    require(manifest["original_metadata_sha256"] == METADATA_SHA, "Recovery original-source link")
    for name, expected in manifest["files"].items():
        require(digest(contents[name]) == expected, "Recovery frozen source/input checksum: " + name)
    require(digest(contents["input/original-final-metadata.zip"]) == METADATA_SHA, "Original metadata embedded in recovery changed")
    for key in ["model", "prompt", "seed", "generation", "datasets", "extraction", "quality"]:
        require(json.loads(contents["input/config.json"])[key] == json.loads(original["input/config.json"])[key], "Recovery scientific config changed: " + key)
    require(json.loads(contents["job_complete.json"])["total"] == 6729, "Incomplete recovery")
    reproduction = json.loads(contents["reproduction_check.json"])
    require(reproduction["passed"] and len(reproduction["controls"]) == 4, "Recovery reproduction QA failed")
    require(all(c["exact_token_path"] and c["feature_arrays_allclose"] for c in reproduction["controls"]), "Reproduction controls changed")
    require(json.loads(contents["quality_gate.json"])["passed"], "Recovery feature QA gate")
    plan = json.loads(contents["input/recovery_plan.json"])
    by_name = {r["name"]: r for name, data in contents.items()
        if name.startswith("outbox/batch_") and name.endswith(".json") for r in [json.loads(data)]}
    required = {r["batch"] for r in plan["missing_batches"]}
    preserved = {r["batch"] for r in plan["preserved"]}
    require(len(by_name) == len(required) == 421 and set(by_name) == required, "Recovery receipt coverage")
    require(len(preserved) == 109 and not preserved & required, "Preserved/recovered overlap")
    require(preserved | required == {r["name"] for r in receipts}, "Combined batch coverage")
    original_questions = {r["id"]: r for r in json.loads(original["input/questions.json"])}
    recovered_questions = json.loads(contents["input/questions.json"])
    planned_ids = [i for b in plan["missing_batches"] for i in b["ids"]]
    require([q["id"] for q in recovered_questions] == planned_ids, "Recovery question ordering")
    require(all(q == original_questions[q["id"]] for q in recovered_questions), "Recovery question content changed")
    result = []
    for old_receipt, old_path in selected:
        name = old_receipt["name"]
        if name in preserved:
            require(old_path.exists(), "Preserved original missing")
            result.append((old_receipt, old_path))
        else:
            require(not old_path.exists(), "Recovery conflicts with newly found original: inspect before publishing")
            new = by_name[name]
            require(new["fingerprint"] == RECOVERY_FINGERPRINT and new["ids"] == old_receipt["ids"], "Recovery batch mismatch")
            result.append((new, recovery / "incoming" / (name + ".zip")))
    return result, contents


def compact_arrays(data, prefix_layers=(6, 9, 11, 12, 18, 24)):
    times = data["checkpoints"]
    layer_indices = [prefix_layers.index(layer) for layer in READOUT_LAYERS]
    result = dict(checkpoints=times.copy(), hidden=data["hidden"][:, RAW_LAYERS].copy(),
        trajectory=data["trajectory"][times].copy(),
        readouts=data["prefix_readouts"][:, layer_indices][:, :, [1, 2, 3]].copy())
    require(result["hidden"].shape == (len(times), 4, 2048), "Unexpected hidden axes")
    require(result["trajectory"].shape == (len(times), 24, 5), "Unexpected trajectory axes")
    require(result["readouts"].shape[1:] == (3, 3), "Unexpected readout axes")
    return result


def combine_examples(examples):
    ids, times, hidden, trajectory, readouts, records, offsets = [], [], [], [], [], [], [0]
    for record, f in examples:
        records.append(record)
        ids.extend([record["id"]] * len(f["checkpoints"]))
        times.append(f["checkpoints"])
        hidden.append(f["hidden"])
        trajectory.append(f["trajectory"])
        readouts.append(f["readouts"])
        offsets.append(offsets[-1] + len(f["readouts"]))
    stream = io.BytesIO()
    expected = dict(ids=np.asarray(ids), checkpoints=np.concatenate(times), hidden=np.concatenate(hidden),
        trajectory=np.concatenate(trajectory), readout_ids=np.asarray([r["id"] for r in records]),
        readout_offsets=np.asarray(offsets, np.int64), readouts=np.concatenate(readouts))
    np.savez_compressed(stream, **expected)
    with np.load(io.BytesIO(stream.getvalue()), allow_pickle=False) as z:
        require(set(z.files) == COMPACT_KEYS, "Compact contains extra features")
        require(z["readout_offsets"][-1] == len(z["readouts"]), "Ragged offset mismatch")
        for key, value in expected.items():
            np.testing.assert_array_equal(z[key], value, err_msg="Compact write/readback: " + key)
    return stream.getvalue(), "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in records).encode("utf-8")


def validate_full(data, record):
    require(set(data.files) == FULL_KEYS, "Unexpected/missing full feature arrays")
    n = len(record["generation"]["generated_token_ids"])
    expected_times = np.array([t for t in CHECKPOINTS if t < n], np.int16)
    np.testing.assert_array_equal(data["checkpoints"], expected_times)
    c = len(expected_times)
    shapes = dict(hidden=(c, 25, 2048), pooled_hidden=(24, 3, 2048), trajectory=(n, 24, 5),
        checkpoint_readouts=(c, 24, 7), checkpoint_top_ids=(c, 24, 10), checkpoint_top_logprob=(c, 24, 10),
        prefix_readouts=(min(n, 129), 6, 5), token_readouts=(n, 5), token_top_ids=(n, 10),
        token_top_logprob=(n, 10), token_region=(n,), token_offsets=(n, 2), token_masks=(n, 3), confidence=(26,), length=(4,))
    for key, shape in shapes.items():
        value = data[key]
        require(value.shape == shape, "Unexpected shape: " + key)
        if key == "confidence":
            require(not np.isinf(value).any(), "Infinite confidence")
        else:
            require(np.isfinite(value).all(), "Nonfinite " + key)
    require(data["hidden"].dtype == np.float32 and data["pooled_hidden"].dtype == np.float32, "Raw precision changed")
    np.testing.assert_allclose(data["checkpoint_readouts"][:, -1, 5], 0, atol=1e-6)


def make_schema(qwen_schema):
    schema = json.loads(json.dumps(qwen_schema))
    schema.update(schema_version="smollm2-qwen-equivalent-compact-v1", model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        dataset="GSM8K train or separately packaged SVAMP; never mix without checking model/dataset/id",
        raw_layers=list(RAW_LAYERS), readout_layers=list(READOUT_LAYERS),
        trajectory_layers="1..24; h0 is embedding; early slices use first12 blocks", primary_layer=12,
        layer_mapping={"7": 6, "10": 9, "13": 11, "14": 12},
        equivalence="Same feature definitions, checkpoints and NPZ keys; relative-depth mapping, NOT same literal layers, vector dimensions, values or tokenizer.")
    schema["dimensions"]["D"] = 2048
    schema["dimensions"]["B"] = "normally32; last GSM17 / SVAMP8"
    schema["npz"]["hidden"]["shape"] = ["C", 4, 2048]
    schema["npz"]["trajectory"]["shape"] = ["C", 24, 5]
    schema["npz"]["trajectory"]["meaning"] = "all24 layer transitions at checkpoints; early predictors use first12 only"
    schema["early_feature_allowlist"] = ["hidden layers6/9/11/12 at available t", "trajectory layers1..12 at t",
        "readout entropy/margin/top1 at layers6/9/12, positions0..t only", "scalar summaries derived from those readouts only"]
    schema["forbidden_predictors"] = ["labels/gold/reference", "ID/dataset/origin as predictor", "total generated length",
        "parse/truncation/analysis eligibility", "future tokens/times", "layers above12 for midlayer prediction", "quality diagnostics"]
    schema["derived_views"]["scalar_column_order"] = list(SCALAR_NAMES)
    schema["id_join_key"] = "dataset + id; never align by row order or join to Qwen labels"
    schema["generation_record"]["generation"]["generated_token_ids"] = "this model's own path; GSM8K greedy, SVAMP sampled"
    schema["generation_record"]["summary"]["origin"] = "preserved_original or recovery_reinference; provenance, NOT split/predictor"
    schema["questions_record"] = dict(id="join key", dataset="gsm8k or svamp", index="source dataset_index",
        question="original problem", gold="reference solution, NEVER a predictor", label="retrospective numeric label",
        question_group_sha256="normalized question group", response_sha256="this model's response hash")
    schema["derived_views"]["problem_index.jsonl"] = "id,dataset,index,question,gold,label,question_group_sha256,response_sha256; metadata/targets, NOT predictors"
    schema["derived_views"]["scalar_features_t0_t32.jsonl"] = "id,dataset,features.t0 (36 floats),features.t32 (36 floats or null)"
    schema["derived_views"]["samples.json"] = "three original generation records per dataset, with question/gold; no raw hidden vectors"
    schema["not_saved"] = ["Compact excludes generated-token logprob/rank, JS, top10, pooled hidden and other full-only features",
        "Neither edition saves full vocabulary logits, full-token dense residual vectors, attention, gradients, trained probes or official GSM8K test outputs"]
    schema["split_policy"] = "No fitted probe or split shipped. Group by normalized question before splitting. Official GSM8K test is excluded. Dataset/model labels must not be mixed."
    return schema


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--recovery", type=Path, help="Complete verified missing-only recovery backup")
    args = p.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    docs = Path(__file__).resolve().parent
    recovery = args.recovery.resolve() if args.recovery else None
    require(not output.is_relative_to(source) and not output.is_relative_to(docs.parents[2])
        and (recovery is None or not output.is_relative_to(recovery)), "Unsafe output directory")
    metadata_zip = source / "final-metadata.zip"
    contents = read_snapshot(metadata_zip, METADATA_SHA)
    run = json.loads(contents["RUN_MANIFEST.json"])
    require(run["fingerprint"] == FINGERPRINT, "Wrong source run")
    config = json.loads(contents["input/config.json"])
    require(config["model"]["expected_layers"] == 24 and config["model"]["expected_hidden_size"] == 2048,
        "Unexpected model architecture")
    require(config["model"]["revision"] == run["model_revision"], "Model revision mismatch")
    require(config["extraction"]["readout_layers"] == [6, 9, 11, 12, 18, 24], "Prefix layer mapping mismatch")
    patch = json.loads(contents["runtime_patch.json"])
    require(digest(contents[patch["source"]]) == patch["source_sha256"], "Runtime patch checksum")
    receipts = [json.loads(data) for name, data in sorted(contents.items()) if name.startswith("outbox/batch_") and name.endswith(".json")]
    require(len(receipts) == 530 and sum(r["count"] for r in receipts) == 8473, "Incomplete final source receipts")
    selected, recovery_contents = recovery_inputs(source, contents, receipts, recovery)
    missing = [r["name"] for r, path in selected if not path.exists()]
    print(json.dumps(dict(required_batches=530, available_batches=530-len(missing), missing_batches=missing)), flush=True)
    if args.preflight:
        return
    require(not missing, f"Cannot build FULL release: {len(missing)} source batches not backed up; no partial release produced")
    require(not output.exists(), "Use a new output directory; do not overwrite a release")
    license_files = {name: (docs.parent / "posthoc-transfer-data-v1" / name).read_bytes()
        for name in ["LICENSE-GSM8K.txt", "LICENSE-SVAMP.txt"]}
    label_answer = runpy.run_path(str(source / "final/src/gsm8k_lens/answers.py"))["label_answer"]
    require(sha256(source / "final/src/gsm8k_lens/answers.py") == run["files"]["src/gsm8k_lens/answers.py"], "Label parser mismatch")
    qwen_schema = json.loads((docs.parent / "DATA_SCHEMA.json").read_text(encoding="utf-8"))
    schema = make_schema(qwen_schema)
    output.mkdir(parents=True)
    (output / "COMPACT_SCHEMA.json").write_bytes(encode(schema))
    full_schema = json.loads(contents["feature_schema.json"])
    full_schema["checkpoint_hidden_axes"] = ["available t", "embedding0 then blocks1..24", "hidden2048"]
    full_schema["dimensions"] = dict(T="generated tokens in one response", C="available checkpoints in one response", P="min(T,129)")
    full_schema["prefix_layers"] = [6, 9, 11, 12, 18, 24]
    full_schema["arrays"] = dict(record="Unicode JSON scalar; includes fingerprint/provenance and current label",
        checkpoints=["C"], hidden=["C", 25, 2048], pooled_hidden=[24, 3, 2048], trajectory=["T", 24, 5],
        checkpoint_readouts=["C", 24, 7], checkpoint_top_ids=["C", 24, 10], checkpoint_top_logprob=["C", 24, 10],
        prefix_readouts=["P", 6, 5], token_readouts=["T", 5], token_top_ids=["T", 10], token_top_logprob=["T", 10],
        token_region=["T"], token_offsets=["T", 2], token_masks=["T", 3], confidence=[26], length=[4])
    (output / "FULL_SCHEMA.json").write_bytes(encode(full_schema))
    shutil.copyfile(metadata_zip, output / "smollm2-run-metadata.zip")
    if recovery_contents is not None:
        shutil.copyfile(recovery / "final-metadata.zip", output / "smollm2-recovery-metadata.zip")
    for name in ["README.md", "load_data.py"]:
        shutil.copyfile(docs / name, output / ("DATA_README.md" if name == "README.md" else name))
    buffers = {name: [] for name in ["gsm8k", "svamp"]}
    compact_hashes, compact_numbers = {name: {} for name in buffers}, Counter()
    compact_zips = {name: zipfile.ZipFile(output / ("smollm2-" + name + "-compact-v1.zip"), "w", allowZip64=True) for name in buffers}
    shard = None
    shard_number, shard_bytes, shard_entries = 0, 0, {}
    full_index, seen, stats, checkpoint_counts, index, scalar_rows, samples = [], set(), {}, {}, [], [], []

    def add_compact(dataset, name, content):
        compact_hashes[dataset][name] = digest(content)
        compact_zips[dataset].writestr("smollm2-" + dataset + "-compact-v1/" + name, content,
            compress_type=zipfile.ZIP_STORED if name.endswith(".npz") else zipfile.ZIP_DEFLATED)

    def flush_compact(dataset):
        if not buffers[dataset]:
            return
        npz, text = combine_examples(buffers[dataset])
        name = f"batch_{compact_numbers[dataset]:04d}"
        add_compact(dataset, name + "/features.npz", npz)
        add_compact(dataset, name + "/generations.jsonl", text)
        compact_numbers[dataset] += 1
        buffers[dataset].clear()

    def close_shard():
        nonlocal shard
        if shard is not None:
            shard.writestr("smollm2-full-v1/part-" + f"{shard_number:03d}" + "-README.md", (docs / "README.md").read_bytes(), compress_type=zipfile.ZIP_DEFLATED)
            shard.writestr("smollm2-full-v1/part-" + f"{shard_number:03d}" + "-manifest.json", encode(shard_entries), compress_type=zipfile.ZIP_DEFLATED)
            shard.close()
            shard = None

    try:
        for receipt, path in selected:
            require(path.stat().st_size == receipt["bytes"] and sha256(path) == receipt["archive_sha256"], "Source ZIP mismatch: " + path.name)
            with zipfile.ZipFile(path) as original:
                require(set(original.namelist()) == set(receipt["members"]), "Unexpected source members")
                for name, expected in receipt["members"].items():
                    require(digest(original.read(name)) == expected, "Source member checksum")
                records = [json.loads(line) for line in original.read("generations.jsonl").splitlines()]
                require([r["id"] for r in records] == receipt["ids"], "Source row order mismatch")
                for record in records:
                    identifier, dataset = record["id"], record["dataset"]
                    require(identifier not in seen and record["fingerprint"] == receipt["fingerprint"], "Duplicate ID/wrong run")
                    require(record["fingerprint"] in [FINGERPRINT, RECOVERY_FINGERPRINT], "Unknown run")
                    origin = "recovery_reinference" if record["fingerprint"] == RECOVERY_FINGERPRINT else "preserved_original"
                    if origin == "recovery_reinference":
                        require(record["provenance"]["kind"] == origin and record["provenance"]["original_batch"] == receipt["name"], "Missing recovery provenance")
                    seen.add(identifier)
                    label = label_answer(record["generation"]["text"], record["gold"], record["generation"]["truncated"])
                    require(label == record["label"], "Numerical label mismatch")
                    with np.load(io.BytesIO(original.read("features/" + identifier + ".npz")), allow_pickle=False) as data:
                        require(json.loads(str(data["record"])) == record, "NPZ/JSONL record mismatch")
                        validate_full(data, record)
                        f = compact_arrays(data)
                    gen = {k: v for k, v in record["generation"].items() if k != "generation_quality"}
                    gen["effective_stopping_rule"] = "eos_or_completed_final_answer_line_v1"
                    compact_record = dict(id=identifier, dataset=dataset, model_id=config["model"]["id"],
                        model_revision=run["model_revision"], fingerprint=record["fingerprint"], provenance=dict(kind=origin),
                        generation=gen, label=label, summary=dict(id=identifier, origin=origin, incorrect=int(not label["answer_correct"]),
                            parse_status=label["parse_status"], truncated=label["truncated"], generated_tokens=len(gen["generated_token_ids"]), quality=record["quality"]))
                    buffers[dataset].append((compact_record, f))
                    if len(buffers[dataset]) == 32:
                        flush_compact(dataset)
                    counters = stats.setdefault(dataset, Counter())
                    counters.update(answers=1, correct=int(label["answer_correct"]), wrong=int(not label["answer_correct"]),
                        parse_failures=int(label["parse_status"] != "ok"), truncated=int(label["truncated"]),
                        eligible=int(label["analysis_eligible"]))
                    for t in f["checkpoints"]:
                        checkpoint_counts.setdefault(dataset, Counter())[str(int(t))] += 1
                    index.append(dict(id=identifier, dataset=dataset, index=record["dataset_index"], question=record["question"], gold=record["gold"], label=label,
                        source_fingerprint=record["fingerprint"], provenance=origin,
                        question_group_sha256=digest(" ".join(record["question"].split()).encode()),
                        response_sha256=digest(gen["text"].encode())))
                    scalar_rows.append(dict(id=identifier, dataset=dataset, features={
                        "t" + str(t): scalar_features(f["readouts"][:t + 1]).tolist() if len(f["readouts"]) > t else None for t in [0, 32]}))
                    if len([r for r in samples if r["dataset"] == dataset]) < 3:
                        samples.append(dict(**compact_record, question=record["question"], gold=record["gold"]))
            if shard is None or shard_bytes + path.stat().st_size > 960 * 1024**2:
                close_shard()
                shard_number += 1
                shard = zipfile.ZipFile(output / f"smollm2-full-part-{shard_number:03d}.zip", "w", zipfile.ZIP_STORED, allowZip64=True)
                shard_bytes, shard_entries = 0, {}
            member = "smollm2-full-v1/batches/" + path.name
            shard.write(path, member)
            shard_bytes += path.stat().st_size
            entry = dict(batch=path.name, sha256=receipt["archive_sha256"], bytes=receipt["bytes"],
                ids=receipt["ids"], fingerprint=receipt["fingerprint"], part=f"smollm2-full-part-{shard_number:03d}.zip")
            full_index.append(entry)
            shard_entries[member] = entry
            print(receipt["name"], "verified", len(seen), "/8473", flush=True)
        require(len(seen) == 8473 and stats["gsm8k"]["answers"] == 7473 and stats["svamp"]["answers"] == 1000, "Incomplete result")
        for dataset in buffers:
            flush_compact(dataset)
            add_compact(dataset, "DATA_SCHEMA.json", encode(schema))
            add_compact(dataset, "README.md", (docs / "README.md").read_bytes())
            add_compact(dataset, "load_data.py", (docs / "load_data.py").read_bytes())
            questions = [r for r in index if r["dataset"] == dataset]
            add_compact(dataset, "questions.json", encode(questions))
            for name, content in license_files.items():
                add_compact(dataset, name, content)
            compact_zips[dataset].writestr("smollm2-" + dataset + "-compact-v1/checksums.sha256",
                "".join(h + "  " + name + "\n" for name, h in sorted(compact_hashes[dataset].items())), compress_type=zipfile.ZIP_DEFLATED)
    finally:
        close_shard()
        for z in compact_zips.values():
            z.close()
    summary = dict(datasets=stats, available_checkpoints=checkpoint_counts, full_batches=530, full_parts=shard_number,
        published_full_batch_bytes=sum(r["bytes"] for r, _ in selected), compact_raw_layers=list(RAW_LAYERS),
        compact_readout_layers=list(READOUT_LAYERS), model_revision=run["model_revision"],
        numerical_labels_recomputed=8473, source_feature_arrays_validated=8473, packaging_performed_inference_or_training=False)
    summary["provenance_counts"] = dict(Counter(r["provenance"] for r in index))
    summary["source_fingerprints"] = sorted({r["source_fingerprint"] for r in index})
    summary["recovery_was_rerun_inference"] = recovery_contents is not None
    old_labels = {r["id"]: r["label"] for r in json.loads(contents["results/labels.json"])}
    transitions = Counter()
    for r in index:
        transitions[str(old_labels[r["id"]]["answer_correct"]) + "->" + str(r["label"]["answer_correct"])] += 1
    summary["original_vs_published_correctness_transitions"] = dict(transitions)
    (output / "dataset_summary.json").write_bytes(encode(summary))
    (output / "FULL_INDEX.json").write_bytes(encode(dict(parts=shard_number, batches=full_index, source_batch_zip_bytes_preserved=True)))
    published = "# Verified Release Scope\n\nThis package contains all 8,473 responses: "
    published += f"{summary['provenance_counts'].get('preserved_original', 0):,} preserved originals and {summary['provenance_counts'].get('recovery_reinference', 0):,} explicitly regenerated recovery responses.\n"
    published += "Recovery outputs are not the lost original bytes. Do not use the old original numerical labels for regenerated responses. Use each record's current label.\n\n"
    published += "Published counts (not historical counts):\n\n| Dataset | Responses | Correct | Wrong | Format failures |\n|---|---:|---:|---:|---:|\n"
    for domain, count in stats.items():
        published += f"| {domain} | {count['answers']} | {count['correct']} | {count['wrong']} | {count['parse_failures']} |\n"
    published += "\nSee DATA_README.md for definitions and the historical extraction/backup incident. SHA256SUMS.txt covers the actual assets.\n"
    (output / "RELEASE_SCOPE.md").write_text(published, encoding="utf-8")
    with zipfile.ZipFile(output / "AI_STARTER.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in ["RELEASE_SCOPE.md", "DATA_README.md", "COMPACT_SCHEMA.json", "FULL_SCHEMA.json", "dataset_summary.json", "load_data.py"]:
            z.write(output / name, "AI_STARTER/" + name)
        z.writestr("AI_STARTER/problem_index.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in index))
        z.writestr("AI_STARTER/scalar_features_t0_t32.jsonl", "".join(json.dumps(r, allow_nan=False) + "\n" for r in scalar_rows))
        z.writestr("AI_STARTER/samples.json", encode(samples))
    # Read back ALL ZIPs before creating the publication-ready marker.
    for path in output.glob("*.zip"):
        with zipfile.ZipFile(path) as z:
            require(z.testzip() is None, "ZIP readback failed: " + path.name)
            for entry in (e for e in full_index if e["part"] == path.name):
                with z.open("smollm2-full-v1/batches/" + entry["batch"]) as stream:
                    h = hashlib.sha256()
                    for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
                        h.update(chunk)
                require(h.hexdigest() == entry["sha256"], "Full shard did not preserve original ZIP bytes")
        require(path.stat().st_size < 2 * 1024**3, "GitHub asset exceeds 2GiB")
    files = sorted(p for p in output.iterdir() if p.is_file())
    hashes = {p.name: sha256(p) for p in files}
    (output / "SHA256SUMS.txt").write_text("".join(h + "  " + name + "\n" for name, h in hashes.items()), encoding="ascii")
    (output / "READY_TO_PUBLISH.json").write_bytes(encode(dict(complete=True, source_fingerprint=FINGERPRINT,
        recovery_fingerprint=RECOVERY_FINGERPRINT if recovery else None,
        examples=8473, full_original_batches=530, asset_sha256=hashes)))
    print("COMPLETE_RELEASE_READY", flush=True)


if __name__ == "__main__":
    main()
