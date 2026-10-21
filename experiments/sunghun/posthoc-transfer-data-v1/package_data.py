"""Audit saved experiments and package data without inference or fitting."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from sklearn.metrics import roc_auc_score


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_module(path):
    spec = importlib.util.spec_from_file_location("saved_answer_parser", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    outputs = workspace / "outputs"
    out = outputs / "posthoc-transfer-release-v1"
    assets = out / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    docs = Path(__file__).resolve().parent
    roots = {
        "gsm8k": outputs / "posthoc-verifier-v1-r2-backup/snapshots/final",
        "svamp": outputs / "svamp-transfer-v1-backup/snapshots/final",
        "reverse": outputs / "svamp-to-gsm-v1-backup/snapshots/final",
    }
    source_hashes = {}

    def source_bytes(path):
        data = path.read_bytes()
        source_hashes[path.relative_to(workspace).as_posix()] = digest(data)
        return data

    def copy(source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source_bytes(source))

    verified = {}
    for name, root in roots.items():
        manifest = read(root / "snapshot_manifest.json")
        for relative, expected in manifest.items():
            assert digest(source_bytes(root / relative)) == expected, (name, relative)
        verified[name] = len(manifest)
    print("Verified all three source snapshot manifests", verified, flush=True)

    original = outputs / "online-risk-v2-backup"
    questions = {q["id"]: q for q in read(original / "snapshots/final/questions.json")}
    svamp_path = outputs / "svamp-transfer-v1-prepared/input/SVAMP.json"
    svamp_original = source_bytes(svamp_path)
    provenance = read(roots["svamp"] / "input/dataset_provenance.json")
    assert digest(svamp_original) == provenance["original_sha256"]
    svamp_questions = {"svamp-" + q["ID"]: q for q in json.loads(svamp_original)}
    svamp_prompts = {q["id"]: q["question"] for q in read(roots["svamp"] / "input/questions.json")}
    folds = read(roots["reverse"] / "input/provenance.json")["source_folds"]
    labels = {
        "gsm8k": read(roots["gsm8k"] / "input/labels.json"),
        "svamp": read(roots["svamp"] / "results/labels.json"),
    }
    raw = {"gsm8k": {}, "svamp": {}}
    for path in sorted((original / "incoming").glob("batch_*.zip")):
        receipt = read(path.with_suffix(".json"))
        assert digest(source_bytes(path)) == receipt["archive_sha256"]
        with zipfile.ZipFile(path) as z:
            member = path.stem + "/generations.jsonl"
            content = z.read(member)
            assert digest(content) == receipt["files"][member]
            for line in content.splitlines():
                row = json.loads(line)
                assert row["id"] not in raw["gsm8k"]
                raw["gsm8k"][row["id"]] = row
    for path in sorted((roots["svamp"] / "raw").glob("batch_*.jsonl")):
        for line in source_bytes(path).splitlines():
            row = json.loads(line)
            assert row["id"] not in raw["svamp"]
            raw["svamp"][row["id"]] = row

    answer_parser = load_module(roots["gsm8k"] / "src/answers.py")
    all_records, arrays, packages, summaries = {}, {}, {}, {}
    names = read(roots["gsm8k"] / "compact/batch_0000.json")["confidence_names"]
    for dataset, count in [("svamp", 1000), ("gsm8k", 4000)]:
        root = roots[dataset]
        package = out / ("svamp-inference-v1" if dataset == "svamp" else "gsm8k-sampled-posthoc-v1")
        package.mkdir(parents=True, exist_ok=True)
        packages[dataset] = package
        chunks = {}
        details = []
        for path in sorted((root / "compact").glob("batch_*.npz")):
            with np.load(path, allow_pickle=False) as data:
                meta = read(path.with_suffix(".json"))
                assert data["ids"].tolist() == meta["ids"]
                if "confidence_names" in meta:
                    assert meta["confidence_names"] == names
                for key in data.files:
                    chunks.setdefault(key, []).append(data[key])
                details.extend(dict(id=i, **d) for i, d in zip(meta["ids"], meta["details"], strict=True))
        data = {key: np.concatenate(value) for key, value in chunks.items()}
        ids = data["ids"].tolist()
        assert len(ids) == len(set(ids)) == len(labels[dataset]) == len(raw[dataset]) == count
        by_label = {r["id"]: r for r in labels[dataset]}
        records = []
        for identifier in ids:
            label, generation = by_label[identifier], raw[dataset][identifier]["generation"]
            qid = label["question_id"]
            if dataset == "gsm8k":
                question, reference = questions[qid]["question"], questions[qid]["gold"]
                equation = None
                fold = label["fold"]
            else:
                q = svamp_questions[qid]
                question = q["Body"].strip() + " " + q["Question"].strip()
                assert question == svamp_prompts[qid]
                reference, equation, fold = "#### " + str(q["Answer"]), q["Equation"], folds[qid]
            computed = answer_parser.label_answer(generation["text"], reference, generation["truncated"])
            assert all(label[k] == v for k, v in computed.items()), identifier
            records.append(dict(**label, **({"fold": fold} if "fold" not in label else {}),
                                dataset=dataset, question=question, reference=reference,
                                reference_equation=equation, response=generation["text"],
                                response_sha256=digest(generation["text"].encode("utf-8")),
                                generation=generation))
        for key in ["question", "end", "mean", "number"]:
            assert data[key].shape == (count, 1536) and data[key].dtype == np.float32
            assert np.isfinite(data[key]).all()
        assert data["confidence"].shape == (count, 26) and not np.isinf(data["confidence"]).any()
        assert data["length"].shape == (count, 4) and np.isfinite(data["length"]).all()
        for index, row in enumerate(records):
            generation, detail = row["generation"], details[index]
            assert detail["id"] == row["id"]
            expected_length = np.log1p([len(generation["prompt_token_ids"]), len(generation["generated_token_ids"]),
                                       detail["whole_tokens"], detail["number_tokens"]]).astype(np.float32)
            np.testing.assert_array_equal(data["length"][index], expected_length)
        eligible = np.asarray([r["analysis_eligible"] for r in records])
        if "eligible" in data:
            assert np.array_equal(eligible, data["eligible"])
        by_question = {}
        for row in records:
            by_question.setdefault(row["question_id"], set()).add(row["fold"])
        assert all(len(v) == 1 for v in by_question.values())
        (package / "records.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        (assets / (dataset + "_answers.jsonl")).write_text("".join(
            json.dumps({k: v for k, v in r.items() if k != "generation"}, ensure_ascii=False) + "\n"
            for r in records), encoding="utf-8")
        np.savez_compressed(package / "features.npz", **data)
        write(package / "feature_metadata.json", dict(confidence_names=names, details=details))
        copy(root / "input/original_config.json", package / "original_config.json")
        copy(root / "src/config.json", package / "study_config.json")
        if dataset == "svamp":
            (package / "SVAMP.json").write_bytes(svamp_original)
            copy(root / "input/dataset_provenance.json", package / "dataset_provenance.json")
        all_records[dataset], arrays[dataset] = records, data
        summaries[dataset] = dict(answers=count, questions=len(by_question),
            correct=sum(r["answer_correct"] for r in records),
            wrong=sum(not r["answer_correct"] for r in records),
            eligible=int(eligible.sum()), eligible_wrong=sum(not r["answer_correct"] and r["analysis_eligible"] for r in records),
            parse_failures=sum(r["parse_status"] != "ok" for r in records), truncated=sum(r["truncated"] for r in records),
            question_folds=dict(Counter(next(iter(v)) for v in by_question.values())),
            confidence_nan_cells=int(np.isnan(data["confidence"]).sum()))
        print(dataset, summaries[dataset], flush=True)

    annotation = [json.loads(line) for line in (outputs / "error-taxonomy-full-v1/labels.jsonl").read_text(encoding="utf-8").splitlines()]
    records_by_id = {r["id"]: r for rows in all_records.values() for r in rows}
    for row in annotation:
        original_row = records_by_id[row["id"]]
        for field in ["response_sha256", "response", "answer_correct", "fold", "question_id"]:
            assert original_row[field] == row[field], (row["id"], field)
    assert len(annotation) == 1100

    # The published matrices must also match the actual reverse-training inputs.
    for dataset, filenames in [("svamp", ["source_fit_validation.npz", "source_test.npz"]),
                               ("gsm8k", ["target_test.npz"])]:
        lookup = {identifier: i for i, identifier in enumerate(arrays[dataset]["ids"].tolist())}
        for filename in filenames:
            with np.load(roots["reverse"] / "input" / filename, allow_pickle=False) as data:
                order = [lookup[identifier] for identifier in data["ids"].tolist()]
                for key in ["question", "end", "mean", "number", "confidence", "length"]:
                    np.testing.assert_array_equal(arrays[dataset][key][order], data[key])

    transfer = out / "transfer-results-v1"
    transfer.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for name, root in roots.items():
        folder = transfer / name
        for path in sorted((root / "results").glob("*")):
            if path.is_file() and path.suffix in {".json", ".md", ".npz"}:
                copy(path, folder / "results" / path.name)
        model_root = root / "models" if name != "gsm8k" else root / "results"
        for path in model_root.glob("*"):
            if path.is_file() and (path.suffix == ".pt" or
                                  path.suffix == ".json" and (name != "gsm8k" or path.name == "FROZEN_SELECTION.json")):
                copy(path, folder / "models" / path.name)
        copy(root / "src/config.json", folder / "study_config.json")
        if name == "reverse":
            copy(root / "input/provenance.json", folder / "provenance.json")
        # Saved scientific helpers only; deployment/SSH scripts and caches are excluded.
        for filename in ["README.md", "extract.py", "gsm_extract.py", "answers.py", "common.py", "train.py", "gsm_probes.py", "frozen.py", "infer.py", "evaluate.py", "train_source.py", "selection_baselines.py"]:
            if (root / "src" / filename).is_file():
                copy(root / "src" / filename, folder / "methods" / filename)
        metrics[name] = read(root / "results/metrics.json")

    auc_checks = []
    for name, root in roots.items():
        prediction_files = [("source_test", "source_test_predictions.npz"), ("target_test", "target_test_predictions.npz")] if name == "reverse" else [("test", "predictions.npz")]
        for domain, filename in prediction_files:
            with np.load(root / "results" / filename, allow_pickle=False) as p:
                mask = p["eligible"].copy()
                if name == "gsm8k":
                    mask &= p["fold"] == "test"
                for group in ["question", "length", "confidence", "confidence_length", "completed_hidden", "combined", "combined_question"]:
                    actual = roc_auc_score(p["wrong"][mask], p[group][mask])
                    expected = metrics[name]["domains"][domain]["probes"][group]["raw"]["auroc"] if name == "reverse" else metrics[name]["single_answer"][group]["raw"]["auroc"]
                    assert abs(actual - expected) < 1e-12, (name, domain, group)
                    auc_checks.append(dict(study=name, domain=domain, feature=group, auroc=actual))
                relevant = all_records["gsm8k" if name == "gsm8k" or domain == "target_test" else "svamp"]
                lookup = {r["id"]: r for r in relevant}
                for index, identifier in enumerate(p["ids"].tolist()):
                    assert int(not lookup[identifier]["answer_correct"]) == p["wrong"][index]
                    assert lookup[identifier]["analysis_eligible"] == bool(p["eligible"][index])

    schema = dict(version="posthoc-transfer-data-v1", response_key="id", question_group="question_id",
        record_order="features.npz ids equals records.jsonl line order; always check IDs",
        arrays={k: {"shape": ["N", 1536], "dtype": "float32", "layer": 28, "normalization": "before final RMSNorm"} for k in ["question", "end", "mean", "number"]},
        confidence=dict(shape=["N", 26], dtype="float32", names=names, missing="NaN retained for empty number masks; fit-only imputation required"),
        length=dict(shape=["N", 4], dtype="float32", transform="natural log1p", columns=["prompt_tokens", "generated_tokens", "whole_mask_tokens", "number_mask_tokens"]),
        score_note="SVAMP feature score_* and prediction group arrays are uncalibrated mean sigmoid outputs of GSM-trained probes; wrong=1. Reverse prediction arrays use SVAMP-trained probes.",
        source_fold_note="GSM8K folds from original 500-question protocol; SVAMP folds added later for reverse transfer, not used for forward target adaptation.",
        annotation_release="error-taxonomy-full-v1-draft", annotation_semantics_human_validated=False,
        labels="original strict Decimal numerical equality; not human-reviewed reasoning correctness")
    write(assets / "DATA_SCHEMA.json", schema)
    write(assets / "dataset_summary.json", summaries)
    audit = dict(status="passed", source_snapshot_manifest_files_verified=verified,
                 labels_recomputed=5000, response_and_feature_ids_unique=True, shapes_and_finite_values_checked=True,
                 confidence_nans_preserved=True, annotation_overlay_exact_matches=1100,
                 question_folds_consistent=True, stored_aurocs_recomputed=auc_checks,
                 semantic_labels_human_reviewed=False, new_generation_or_training=False)
    write(assets / "RELEASE_AUDIT.json", audit)
    write(transfer / "SOURCE_HASHES.json", source_hashes)
    print("Validated original labels, overlay linkage and all 28 AUROCs", flush=True)
    for package in [*packages.values(), transfer]:
        for filename in ["README.md", "ATTRIBUTION.md", "LICENSE-GSM8K.txt", "LICENSE-SVAMP.txt"]:
            shutil.copyfile(docs / filename, package / filename)
        for filename in ["DATA_SCHEMA.json", "dataset_summary.json", "RELEASE_AUDIT.json"]:
            shutil.copyfile(assets / filename, package / filename)
        payload = sorted(p for p in package.rglob("*") if p.is_file() and p.name != "checksums.sha256")
        (package / "checksums.sha256").write_text("".join(digest(p.read_bytes()) + "  " + p.relative_to(package).as_posix() + "\n" for p in payload), encoding="ascii")
        with zipfile.ZipFile(assets / (package.name + ".zip"), "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for path in sorted(p for p in package.rglob("*") if p.is_file()):
                z.write(path, (Path(package.name) / path.relative_to(package)).as_posix())
        with zipfile.ZipFile(assets / (package.name + ".zip")) as z:
            assert z.testzip() is None
            for line in z.read(package.name + "/checksums.sha256").decode().splitlines():
                expected, name = line.split("  ", 1)
                assert digest(z.read(package.name + "/" + name)) == expected
    shutil.copyfile(docs / "README.md", assets / "DATA_README.md")
    payload = sorted(p for p in assets.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt")
    (assets / "SHA256SUMS.txt").write_text("".join(digest(p.read_bytes()) + "  " + p.name + "\n" for p in payload), encoding="ascii")
    print(json.dumps({p.name: p.stat().st_size for p in assets.iterdir() if p.is_file()}), flush=True)


if __name__ == "__main__":
    main()
