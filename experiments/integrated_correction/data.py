"""Pin upstream parquet rows, verify release assets, and split exact duplicate groups."""

import gzip
import json
import re
import shutil
import unicodedata
from collections import defaultdict
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import ZipFile

from huggingface_hub import HfApi
import pyarrow.parquet as pq

from .answers import extract
from .storage import digest, index, sha256

DATASETS = {
    "gsm8k": ("openai/gsm8k", ("main",)),
    "math": ("EleutherAI/hendrycks_math", ("algebra", "counting_and_probability", "geometry",
                                          "intermediate_algebra", "number_theory", "prealgebra", "precalculus")),
}
CONSISTENCY_TAG = "gsm8k-consistency-types-v1"
RELEASE_URL = "https://github.com/KyeongwooLee/AI-Endoscope/releases/download"
RELEASES = {
    "gsm8k": ("gsm8k-train-features-v1", "AI_STARTER.zip",
              "FP32, original seed, and final-line stopping differ"),
    "math": ("math-qwen25-response-labels-v1", "qwen25-math-responses-25458-v1.jsonl.gz",
             "different prompt/cohort and exact token IDs/role seeds unavailable"),
}


def fetch(url, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".download")
        with urlopen(url, timeout=45) as incoming, temporary.open("wb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
        temporary.replace(path)
    return path


def release_questions(dataset, directory, explicit):
    tag, name, response_reason = RELEASES[dataset]
    base = f"{RELEASE_URL}/{tag}"
    provenance = {"url": f"{base}/{name}", "generated_responses_reused": 0,
                  "generated_response_compatibility": response_reason}
    try:
        root = directory / tag
        if explicit:
            checksums, asset = root / "SHA256SUMS.txt", root / name
        else:
            checksums = fetch(f"{base}/SHA256SUMS.txt", root / "SHA256SUMS.txt")
            asset = fetch(f"{base}/{name}", root / name)
        expected = {row.split(maxsplit=1)[1].lstrip("*"): row.split()[0]
                    for row in checksums.read_text().splitlines() if row.strip()}
        if sha256(asset) != expected[name]:
            raise ValueError(f"Invalid release checksum: {asset}")
    except URLError as exc:
        if explicit:
            raise
        return {}, {**provenance, "unavailable": str(exc)}
    provenance.update(sha256=expected[name], checksum_file_sha256=sha256(checksums), matched_rows=0)
    if dataset == "gsm8k":
        with ZipFile(asset) as archive:
            release = json.loads(archive.read("AI_STARTER/questions.json"))
        return {(row["question"], row["gold"]): (row["question"], row["gold"]) for row in release}, provenance
    with gzip.open(asset, "rt", encoding="utf-8") as stream:
        return {(row["problem"], row["reference_solution"]): (row["problem"], row["reference_solution"])
                for row in map(json.loads, stream)}, provenance


def consistency_assets(config, cache):
    root = Path(config["release_dir"] or cache / "releases") / CONSISTENCY_TAG
    names = ("SHA256SUMS.txt", "types.jsonl", "greedy_token_ids.jsonl")
    return [root / name if config["release_dir"] else fetch(f"{RELEASE_URL}/{CONSISTENCY_TAG}/{name}", root / name)
            for name in names]


def consistency_rows(config, cache):
    checksums, labels_path, tokens_path = consistency_assets(config, cache)
    expected = {name: value for value, name in (line.split() for line in checksums.read_text().splitlines() if line.strip())}
    actual = {path.name: sha256(path) for path in (labels_path, tokens_path)}
    if any(expected.get(name) != value for name, value in actual.items()):
        raise ValueError("GSM8K consistency release checksum mismatch")
    with labels_path.open(encoding="utf-8") as labels, tokens_path.open(encoding="utf-8") as tokens:
        by_id = index(map(json.loads, labels))
        greedy = index(map(json.loads, tokens))
    if not by_id or by_id.keys() != greedy.keys():
        raise ValueError("GSM8K consistency release label/token IDs differ")
    by_row = {}
    for key, row in by_id.items():
        pid = row["pid"]
        if type(pid) is not int or pid < 0 or key != f"train-{pid}" or pid in by_row:
            raise ValueError(f"Invalid GSM8K release problem ID: {key}")
        samples, valid = row["samples_pred"], row["samples_format_ok"]
        if len(samples) != 30 or len(valid) != 30 or any(type(flag) is not bool for flag in valid):
            raise ValueError(f"Invalid 30-answer bank: {key}")
        token_row = greedy[key]
        for name in ("prompt_ids", "output_ids"):
            if not token_row[name] or any(type(token) is not int or not 0 <= token < 151936 for token in token_row[name]):
                raise ValueError(f"Invalid {name}: {key}")
        by_row[pid] = (row, token_row)
    return by_row, actual


def group_id(question):
    return digest(" ".join(unicodedata.normalize("NFC", question).split()))


def split_problems(rows, seed, limit):
    test_groups = {r["group"] for r in rows if r["official_split"] == "test"}
    removed = [r["id"] for r in rows if r["official_split"] == "train" and r["group"] in test_groups]
    removed_set = set(removed)
    train = defaultdict(list)
    for row in rows:
        if row["official_split"] == "train" and row["id"] not in removed_set:
            train[row["group"]].append(row)

    # A duplicate with differing metadata uses the first stable stratum for the whole group.
    strata = defaultdict(list)
    for group, members in train.items():
        stratum = min((r["dataset"], r["subset"], r["level"] or "") for r in members)
        strata[stratum].append(group)
    for stratum, groups in sorted(strata.items()):
        groups.sort(key=lambda g: digest([seed, stratum, g]))
        n = len(groups)
        for index, group in enumerate(groups):
            fold = "fit" if index < round(n * .8) else "tune" if index < round(n * .9) else "calibration"
            for row in train[group]:
                row["split"] = fold
    for row in rows:
        if row["official_split"] == "test":
            row["split"] = "test"
    retained = [r for r in rows if r["id"] not in removed_set]
    if limit is not None:
        selected = []
        for dataset in DATASETS:
            ranked = sorted((r for r in retained if r["dataset"] == dataset),
                            key=lambda r: digest([seed, "pilot", r["id"]]))
            remaining = limit
            # Include each inner fold when possible and at least one official test problem.
            test_n = max(1, round(limit * .2))
            candidates = {"test": test_n, "fit": max(1, round((limit - test_n) * .8)),
                          "tune": round((limit - test_n) * .1), "calibration": round((limit - test_n) * .1)}
            chosen = set()
            for fold, count in candidates.items():
                for row in [r for r in ranked if r["split"] == fold][:min(remaining, count)]:
                    chosen.add(row["id"])
                    remaining -= 1
            selected.extend([r for r in ranked if r["id"] in chosen] +
                            [r for r in ranked if r["id"] not in chosen][:remaining])
        retained = selected
    validate_partitions(retained)
    return sorted(retained, key=lambda r: r["id"]), sorted(removed)


def validate_partitions(rows):
    index(rows)
    groups = defaultdict(set)
    for row in rows:
        groups[row["group"]].add(row["split"])
        if (row["official_split"] == "test") != (row["split"] == "test"):
            raise ValueError("Official test partition mismatch")
    if any(len(folds) > 1 for folds in groups.values()):
        raise ValueError("Question group leaks across splits")


def prepare(config, cache, pinned=None):
    api = HfApi()
    rows, sources = [], {}
    released, bank = None, None
    if config.get("gsm8k_release_bank"):
        released, hashes = consistency_rows(config, cache)
        bank = {"tag": CONSISTENCY_TAG, "url": f"{RELEASE_URL}/{CONSISTENCY_TAG}",
                "assets_sha256": hashes, "temperature": 1., "top_p": 1.,
                "test_oracle_sampling": {"temperature": 1., "top_p": 1., "engine": "local Transformers"},
                "contents": "answers and format flags only; token IDs for greedy only",
                "unavailable": ["sample_text", "sample_tokens", "sample_finish", "per_response_seed", "refusal"],
                "excluded_train_ids": []}
    for dataset, (repo, subsets) in DATASETS.items():
        if bank and dataset != "gsm8k":
            continue
        revision = (pinned[dataset] if pinned else config[f"{dataset}_revision"])
        if revision is None:
            revision = api.dataset_info(repo).sha
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"Expected a pinned SHA for {dataset}")
        release, provenance = release_questions(dataset, Path(config["release_dir"] or cache / "releases"),
                                                explicit=config["release_dir"] is not None)
        sources[dataset] = {"repo": repo, "revision": revision, "release": provenance, "parquet": {}}
        for subset in subsets:
            for official_split in ("train", "test"):
                name = f"{subset}/{official_split}-00000-of-00001.parquet"
                path = fetch(f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}",
                             cache / dataset / revision / name)
                sources[dataset]["parquet"][name] = sha256(path)
                for raw_index, raw in enumerate(pq.read_table(path).to_pylist()):
                    question, solution = ((raw["question"], raw["answer"]) if dataset == "gsm8k"
                                          else (raw["problem"], raw["solution"]))
                    found = (question, solution) in release
                    if found:
                        question, solution = release[question, solution]
                        provenance["matched_rows"] += 1
                    rows.append({"id": f"{dataset}/{subset}/{official_split}/{raw_index}", "dataset": dataset,
                                 "subset": subset, "level": raw.get("level"), "official_split": official_split,
                                 "raw_index": raw_index, "question": question, "group": group_id(question),
                                 "gold": extract(dataset, solution), "reference_source": "release" if found else "parquet"})
    if bank:
        train = {row["raw_index"]: row for row in rows if row["official_split"] == "train"}
        if released.keys() - train.keys():
            raise ValueError("GSM8K consistency release contains unknown official rows")
        for pid, (label, tokens) in released.items():
            row = train[pid]
            if row["question"] != label["question"] or row["gold"] != extract("gsm8k", label["gsm8k_solution"]):
                raise ValueError(f"GSM8K consistency release disagrees with official row {pid}")
            row["release_bank"] = {"id": label["id"], "samples": label["samples_pred"],
                                   "format_ok": label["samples_format_ok"], "prompt_ids": tokens["prompt_ids"],
                                   "output_ids": tokens["output_ids"]}
        bank["excluded_train_ids"] = [row["id"] for row in train.values() if "release_bank" not in row]
        rows = [row for row in rows if row["official_split"] == "test" or "release_bank" in row]
    rows, removed = split_problems(rows, config["seed"], config["limit_per_dataset"])
    return rows, {"datasets": sources, "gsm8k_release_bank": bank, "removed_train_test_duplicates": removed,
                  "missing_gold": [row["id"] for row in rows if row["gold"] is None]}
