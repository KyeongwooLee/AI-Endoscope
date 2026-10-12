"""Verify immutable backup receipts, then build flat release and small AI bundle."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import runpy
import zipfile

import numpy as np

from load_train_all import CHECKPOINTS, SCALAR_NAMES, scalar_features, sha256

FINGERPRINT = "38d173815f312ea20a45d08cf33fe6795b9411e02c155bbd16079c7fa77d1ea4"
METADATA_SHA = "d766d0e51f6d0640af59b6df19fb1b091589a5dd683b8347558a4abc0c990098"
LICENSE = """MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def encode(value, pretty=True):
    return (json.dumps(value, indent=2 if pretty else None, ensure_ascii=True,
                       allow_nan=False) + "\n").encode("utf-8")


def digest(content):
    return hashlib.sha256(content).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path, help="NEW directory outside the git checkout")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    docs = Path(__file__).resolve().parents[1]
    require(not output.is_relative_to(source), "Do not write into original backup")
    require(not output.is_relative_to(docs.parents[1]), "Do not write release data into git checkout")
    output.mkdir(parents=True, exist_ok=False)
    original_receipt = json.loads((source / "local_backup_receipt.json").read_text())
    require(original_receipt["verified_batches"] == 234, "Incomplete local backup receipt")
    require(sha256(source / "metadata-backup.zip") == METADATA_SHA, "Metadata backup hash mismatch")
    with zipfile.ZipFile(source / "metadata-backup.zip") as archive:
        metadata = {name: archive.read(name) for name in archive.namelist()}
    backup = json.loads(metadata["backup_manifest.json"])
    # Original metadata manifest lists only files present before the manifest itself.
    entries = backup.get("files", backup)
    require(isinstance(entries, dict), "Unknown metadata manifest structure")
    for name, value in entries.items():
        require(name in metadata, f"Missing original metadata member: {name}")
        expected = value.get("sha256") if isinstance(value, dict) else value
        require(digest(metadata[name]) == expected, f"Metadata member hash mismatch: {name}")
    manifest = json.loads(metadata["manifest.json"])
    require(manifest["fingerprint"] == FINGERPRINT, "Protocol mismatch")
    questions = json.loads(metadata["questions.json"])
    q_by_id = {row["id"]: row for row in questions}
    require(len(q_by_id) == 7473, "Question ID count mismatch")
    require(sha256(source / "gsm8k_lens/answers.py") == digest(metadata["gsm8k_lens/answers.py"]),
            "Local label parser differs from immutable backup")
    label_answer = runpy.run_path(str(source / "gsm8k_lens/answers.py"))["label_answer"]
    group_by_id = {key: digest(" ".join(row["question"].split()).encode("utf-8"))
                   for key, row in q_by_id.items()}
    full_zip = output / "gsm8k-train-all-v1.zip"
    index_path = output / "problem_index.jsonl"
    scalar_path = output / "scalar_features_t0_t32.jsonl"
    members = {}
    original_archives = {}
    seen = set()
    counts, origins, checkpoint_counts, checkpoint_wrong = Counter(), {}, Counter(), Counter()
    lengths, samples = [], {}
    prefix = "gsm8k-train-all-v1/"

    with zipfile.ZipFile(full_zip, "w", allowZip64=True) as release, \
            index_path.open("wb") as index_file, scalar_path.open("wb") as scalar_file:
        def add(name, content):
            require(name not in members, f"Duplicate release member: {name}")
            if not name.endswith(".npz"):
                for forbidden in (b"PRIVATE KEY-----", b"ghp_", b"github_pat_", b"165.132.46.",
                                  b"C:\\\\Users\\\\sungh", b"C:/Users/sungh"):
                    require(forbidden not in content, f"Private operational value in {name}")
            method = zipfile.ZIP_STORED if name.endswith(".npz") else zipfile.ZIP_DEFLATED
            release.writestr(prefix + name, content, compress_type=method)
            members[name] = {"sha256": digest(content), "bytes": len(content)}

        for name, content in metadata.items():
            add(name, content)
        for number in range(234):
            name = f"batch_{number:04d}"
            archive_path = source / "incoming" / (name + ".zip")
            receipt_bytes = (source / "incoming" / (name + ".json")).read_bytes()
            receipt = json.loads(receipt_bytes)
            archive_sha = sha256(archive_path)
            require(archive_sha == receipt["archive_sha256"] == original_receipt["batch_hashes"][name],
                    f"Original archive hash mismatch: {name}")
            require(archive_path.stat().st_size == receipt["archive_bytes"], "Original archive size mismatch")
            require(receipt["fingerprint"] == FINGERPRINT, "Batch protocol mismatch")
            require(json.loads(metadata[f"outbox/{name}.json"]) == receipt, "Metadata vs incoming receipt mismatch")
            original_archives[name] = {"sha256": archive_sha, "bytes": archive_path.stat().st_size}
            with zipfile.ZipFile(archive_path) as archive:
                require(set(archive.namelist()) == set(receipt["files"]), "Unexpected batch members")
                contents = {key: archive.read(key) for key in archive.namelist()}
            for key, content in contents.items():
                require(digest(content) == receipt["files"][key], f"Original member mismatch: {key}")
                add(key, content)
            records = [json.loads(line) for line in contents[f"{name}/generations.jsonl"].decode().split("\n") if line]
            require([r["id"] for r in records] == receipt["ids"], "Record ordering mismatch")
            require([r["summary"] for r in records] == json.loads(contents[f"{name}/summary.json"]), "Summary mismatch")
            require(len(records) == (17 if number == 233 else 32), "Unexpected batch size")
            with np.load(io.BytesIO(contents[f"{name}/features.npz"]), allow_pickle=False) as a:
                require(set(a.files) == {"ids", "checkpoints", "hidden", "trajectory", "readout_ids", "readout_offsets", "readouts"}, "NPZ key mismatch")
                c, b, r = len(a["ids"]), len(records), len(a["readouts"])
                require(a["hidden"].shape == (c, 4, 1536), "Hidden shape mismatch")
                require(a["trajectory"].shape == (c, 28, 5), "Trajectory shape mismatch")
                require(a["readouts"].shape == (r, 3, 3), "Readout shape mismatch")
                require(a["checkpoints"].shape == (c,), "Checkpoint shape mismatch")
                require(a["readout_offsets"].shape == (b + 1,), "Offset shape mismatch")
                require(a["readout_offsets"][0] == 0 and a["readout_offsets"][-1] == r, "Offset bounds mismatch")
                require(a["readout_ids"].tolist() == receipt["ids"], "Readout ID mismatch")
                for key in ("hidden", "trajectory", "readouts"):
                    require(a[key].dtype == np.float32 and np.isfinite(a[key]).all(), f"Nonfinite or wrong dtype: {key}")
                require(a["checkpoints"].dtype == np.int16 and a["readout_offsets"].dtype == np.int64, "Index dtype mismatch")
                expected_ids, expected_times = [], []
                for i, record in enumerate(records):
                    key, generation, label = record["id"], record["generation"], record["label"]
                    require(key in q_by_id and key not in seen, "Unknown or duplicate problem ID")
                    require(record["fingerprint"] == FINGERPRINT, "Record protocol mismatch")
                    seen.add(key)
                    require(label_answer(generation["text"], q_by_id[key]["gold"], generation["truncated"]) == label,
                            f"Recomputed label mismatch: {key}")
                    length = generation["generated_tokens"]
                    require(length == len(generation["generated_token_ids"]) and length > 0, "Generation length mismatch")
                    require(generation["prompt_tokens"] == len(generation["prompt_token_ids"]), "Prompt length mismatch")
                    times = [t for t in CHECKPOINTS if t < length]
                    expected_ids.extend([key] * len(times))
                    expected_times.extend(times)
                    start, end = map(int, a["readout_offsets"][i:i + 2])
                    require(end - start == min(129, length), "Readout length mismatch")
                    incorrect = int(not label["answer_correct"])
                    require(record["summary"]["incorrect"] == incorrect, "Incorrect target mismatch")
                    counts.update(problems=1, incorrect=incorrect, correct=1-incorrect,
                                  parse_failures=int(label["parse_status"] != "ok"),
                                  truncated=int(label["truncated"]),
                                  truncated_parse_failures=int(label["truncated"] and label["parse_status"] != "ok"),
                                  parsed_nontruncated=int(label["analysis_eligible"]),
                                  parsed_nontruncated_incorrect=int(label["analysis_eligible"] and incorrect),
                                  replay_positions=record["summary"]["quality"]["positions"],
                                  replay_matches=record["summary"]["quality"]["matches"])
                    origin = record["summary"]["origin"]
                    origins.setdefault(origin, Counter()).update(problems=1, incorrect=incorrect)
                    lengths.append(length)
                    for t in times:
                        checkpoint_counts[t] += 1
                        checkpoint_wrong[t] += incorrect
                    index_row = {"id": key, "batch": name, "question_group_sha256": group_by_id[key],
                                 "origin": origin, "incorrect": incorrect, **{k: label[k] for k in
                                 ("answer_correct", "parse_status", "truncated", "analysis_eligible")},
                                 "generated_tokens": length, "available_checkpoints": times}
                    index_file.write(encode(index_row, False))
                    derived = {"id": key, "incorrect": incorrect, "features": {}}
                    for t in (0, 32):
                        derived["features"][f"t{t}"] = (scalar_features(a["readouts"][start:start+t+1]).tolist()
                                                         if t in times else None)
                    scalar_file.write(encode(derived, False))
                    category = ("parse_failure" if label["parse_status"] != "ok" else
                                "numeric_incorrect" if incorrect else "correct")
                    if category not in samples:
                        samples[category] = {"category": category, "question": q_by_id[key],
                                             "record": record, "derived_scalar": derived}
                require(a["ids"].tolist() == expected_ids, "Checkpoint ID sequence mismatch")
                require(a["checkpoints"].tolist() == expected_times, "Checkpoint time sequence mismatch")
            if number % 25 == 0 or number == 233:
                print(json.dumps({"verified_batches": number+1, "problems": len(seen)}), flush=True)
        require(seen == set(q_by_id) == {f"train-{i:05d}" for i in range(7473)}, "Not all official train IDs")
        require(counts["correct"] == 6091 and counts["incorrect"] == 1382, "Final label count mismatch")
        require(counts["parse_failures"] == 78 and counts["truncated"] == 16, "Final failure count mismatch")
        require(counts["replay_matches"] == counts["replay_positions"] == 28251, "Replay mismatch")
        require(dict(checkpoint_counts) == {0: 7473, 32: 7467, 64: 7446, 128: 5865}, "Checkpoint count mismatch")
        source_summary = json.loads(metadata["results/generation_summary.json"])
        require(source_summary["by_origin"] == origins, "Origin summary mismatch")
        group_sizes = Counter(group_by_id.values())
        summary = {
            "schema_version": "train-compact-release-v1", "release_tag": "gsm8k-train-features-v1",
            "scope": "Official GSM8K train only; not a supplied fit/validation/test split",
            "counts": dict(counts), "accuracy": counts["correct"] / len(seen),
            "checkpoint_problems": dict(checkpoint_counts), "checkpoint_incorrect": dict(checkpoint_wrong),
            "by_origin": origins, "generated_tokens_mean": float(np.mean(lengths)),
            "generated_tokens_median": float(np.median(lengths)),
            "unique_normalized_question_groups": len(group_sizes),
            "duplicate_question_groups": sum(n > 1 for n in group_sizes.values()),
            "duplicate_question_extra_rows": sum(n-1 for n in group_sizes.values()),
            "scalar_feature_names": SCALAR_NAMES, "scalar_checkpoint_positions": [0,32],
            "target": "incorrect=1; correct=0; parse failures included in incorrect",
            "official_test_loaded": False, "probe_retrained_on_full_pool": False,
            "model": manifest["config"]["model"], "dataset": manifest["config"]["dataset"],
            "prompt": manifest["config"]["prompt"], "fingerprint": FINGERPRINT,
            "original_batch_zip_bytes": sum(v["bytes"] for v in original_archives.values()),
            "source_metadata_zip_sha256": METADATA_SHA,
            "warnings": ["Origin names are provenance, NOT split assignments.",
                         "No calibrated probability of answer correctness is provided.",
                         "Exclude labels, gold, lengths, origin/ID and future time/layer values from early predictors.",
                         "Full pool includes previous fit/tuning/evaluation examples.",
                         "Scalar convenience views are new aggregation, not new inference or trained results."]
        }
        (output / "dataset_summary.json").write_bytes(encode(summary))
        (output / "samples.json").write_bytes(encode(list(samples.values())))
        (output / "DATA_README.md").write_bytes((docs / "DATA_README.md").read_bytes())
        (output / "DATA_SCHEMA.json").write_bytes((docs / "DATA_SCHEMA.json").read_bytes())
        add("README.md", (docs / "DATA_README.md").read_bytes())
        add("DATA_SCHEMA.json", (docs / "DATA_SCHEMA.json").read_bytes())
        add("dataset_summary.json", encode(summary))
        add("samples.json", encode(list(samples.values())))
        add("load_train_all.py", (docs / "data_tools/load_train_all.py").read_bytes())
        add("LICENSE-GSM8K.txt", LICENSE.encode())
        attribution = ("# Attribution\n\nGSM8K questions and gold solutions: Copyright (c) 2021 OpenAI.\n"
                       "Source: https://github.com/openai/grade-school-math\n"
                       "License: https://github.com/openai/grade-school-math/blob/master/LICENSE\n"
                       "The upstream MIT notice is reproduced in LICENSE-GSM8K.txt.\n\n"
                       "Model: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct\n"
                       "No model weights are distributed. Generated responses may be incorrect.\n"
                       "This attribution does not relicense the whole repository or the model.\n")
        add("ATTRIBUTION.md", attribution.encode())
        index_file.flush()
        scalar_file.flush()
        add("problem_index.jsonl", index_path.read_bytes())
        add("scalar_features_t0_t32.jsonl", scalar_path.read_bytes())
        audit = {"built_at_utc": datetime.now(timezone.utc).isoformat(),
                 "fingerprint": FINGERPRINT, "verified_original_batches": 234,
                 "verified_problems": len(seen), "recomputed_all_labels": True,
                 "validated_all_npz_shapes_dtypes_finite": True,
                 "original_metadata_sha256": METADATA_SHA,
                 "original_batch_archives_not_included": original_archives,
                 "members_excluding_this_manifest_and_checksums": members.copy()}
        add("release_manifest.json", encode(audit))
        checksums = "".join(f"{value['sha256']}  {name}\n" for name, value in sorted(members.items()))
        release.writestr(prefix + "checksums.sha256", checksums.encode())

    # Read back the newly written archive; verify bytes, not just ZIP metadata.
    with zipfile.ZipFile(full_zip) as archive:
        require(len(archive.namelist()) == len(members)+1, "Release member count mismatch")
        for name, value in members.items():
            require(digest(archive.read(prefix+name)) == value["sha256"], f"Release readback mismatch: {name}")
    print(json.dumps({"full_zip_verified_members": len(members), "bytes": full_zip.stat().st_size}), flush=True)
    starter_files = ["README.md", "DATA_SCHEMA.json", "dataset_summary.json", "samples.json",
                     "questions.json", "problem_index.jsonl", "scalar_features_t0_t32.jsonl",
                     "LICENSE-GSM8K.txt", "ATTRIBUTION.md"]
    with zipfile.ZipFile(full_zip) as full, zipfile.ZipFile(output / "AI_STARTER.zip", "w", zipfile.ZIP_DEFLATED) as starter:
        starter_sums = []
        for name in starter_files:
            content = full.read(prefix+name)
            starter.writestr("AI_STARTER/"+name, content)
            starter_sums.append(f"{digest(content)}  {name}\n")
        starter.writestr("AI_STARTER/checksums.sha256", "".join(starter_sums))
    public_assets = ["gsm8k-train-all-v1.zip", "AI_STARTER.zip", "DATA_README.md", "DATA_SCHEMA.json", "dataset_summary.json"]
    sums = "".join(f"{sha256(output/name)}  {name}\n" for name in public_assets)
    (output / "SHA256SUMS.txt").write_text(sums, encoding="utf-8", newline="\n")
    (output / "packaging_audit.json").write_bytes(encode({"verified_batches":234,
        "verified_problems":len(seen), "verified_release_members":len(members),
        "assets": {name: {"bytes":(output/name).stat().st_size, "sha256":sha256(output/name)} for name in public_assets}}))
    print((output / "packaging_audit.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
