"""Training stages and artifact contracts shared by tuning and final evaluation."""

from collections import defaultdict
from decimal import Decimal

import numpy as np
import torch

from .answers import CLASSES, REFUSAL_VERSION, REFUSALS, distribution, extract, grade
from .data import CONSISTENCY_TAG, prepare, validate_partitions
from .learning import learn_probes, train_reward
from .model import (GSM_SYSTEM, GSM_USER, HIDDEN_SIZE, HOOK, MAX_NEW, MATH_SYSTEM, MATH_USER, TEMPERATURE, TOP_P,
                    Engine, Problem, prefix_positions, response_id, response_seed)
from .storage import digest, index, read, sha256, write


def prepare_stage(stage, config, cache, _deps):
    def build(directory):
        problems, metadata = prepare(config, cache)
        write(directory / "problems.json", problems)
        write(directory / "provenance.json", {
            **metadata, "model": {"repo": config["model"], "revision": config["model_revision"],
                                     "weights_sha256": config["model_weights_sha256"]},
            "generation": {"max_new_tokens": MAX_NEW, "temperature": TEMPERATURE, "top_p": TOP_P,
                           "seed": config["seed"], "rng": "numpy PCG64, one SHA-256-derived seed per response",
                           "eos_or_length": True, "label_bank_samples": 30,
                           "gsm8k": [GSM_SYSTEM, GSM_USER], "math": [MATH_SYSTEM, MATH_USER]},
            "refusals": {"version": REFUSAL_VERSION, "phrases": REFUSALS}, "hook": HOOK,
        })
    stage.unit("all", build)


def problems(deps, test=False):
    rows = read(deps["prepare"].read_unit("all") / "problems.json")
    validate_partitions(rows)
    return [row for row in rows if (row["split"] == "test") == test]


def score(row, response):
    if response.get("problem_id") != row["id"]:
        raise ValueError("Response joined to the wrong problem")
    return {**response, "split": row["split"], "grading": grade(row["dataset"], response, row["gold"])}


def validate_responses(row, responses, seed, role, indices, *, released=False):
    keyed = index(responses)
    expected = {response_id(row["id"], role, i): i for i in indices}
    if keyed.keys() != expected.keys():
        raise ValueError(f"Unexpected {role} IDs for {row['id']}")
    for identifier, i in expected.items():
        response = keyed[identifier]
        if (response["problem_id"], response["split"], response["role"], response["sample_index"], response["seed"]) != (
                row["id"], row["split"], role, i, None if released else response_seed(seed, row["id"], role, i)):
            raise ValueError(f"Response provenance mismatch: {identifier}")
        if released and response.get("source") != CONSISTENCY_TAG:
            raise ValueError(f"Response provenance mismatch: {identifier}")
    return [keyed[response_id(row["id"], role, i)] for i in indices]


def build_bank(row, engine, seed, **sampling):
    p = Problem.from_row(row)
    indices = list(range(30))
    engine.start_cost()
    prompt = engine.prompt(p)
    responses = [score(row, response) for response in
                 engine.generate_many(p, prompt, "label_bank", indices, seed, True, **sampling)]
    validate_responses(row, responses, seed, "label_bank", indices)
    return responses


def release_bank(row):
    source = row["release_bank"]
    responses = []
    for i, (value, formatted) in enumerate(zip(source["samples"], source["format_ok"])):
        answer = None
        if formatted:
            if type(value) not in (float, int) or not Decimal(str(value)).is_finite():
                raise ValueError(f"Invalid formatted answer for {row['id']} sample {i}")
            answer = extract("gsm8k", "#### " + format(Decimal(str(value)), "f"))
        responses.append({"id": response_id(row["id"], "label_bank", i), "problem_id": row["id"],
                          "split": row["split"], "role": "label_bank", "sample_index": i, "seed": None,
                          "source": CONSISTENCY_TAG, "source_id": source["id"], "finish": None,
                          "prompt_ids": None, "output_ids": None, "text": None,
                          "generation": {"temperature": 1., "top_p": 1.},
                          "grading": {"answer": answer, "status": "parsed" if formatted else "release_format_failure",
                                      "correct": answer is not None and answer == row["gold"],
                                      "explicit_refusal": None}})
    return responses


def label_bank_stage(stage, config, _cache, deps):
    engine = None if config.get("gsm8k_release_bank") else Engine.load(config)
    for row in problems(deps):
        def build(out):
            records = release_bank(row) if engine is None else build_bank(row, engine, config["seed"])
            write(out / "responses.json", records)
        stage.unit(row["id"], build)


def load_bank(row, stage, seed):
    path = stage.read_unit(row["id"]) / "responses.json"
    return validate_responses(row, read(path), seed, "label_bank", list(range(30)), released="release_bank" in row)


def distributions_stage(stage, config, _cache, deps):
    for row in problems(deps):
        def build(out):
            responses = load_bank(row, deps["generate-label-bank"], config["seed"])
            write(out / "distribution.json", distribution(row["dataset"], responses))
        stage.unit(row["id"], build)


def save_features(out, arrays, responses, lengths):
    ids = [item["id"] for item in responses]
    path = out / "features.npz"
    np.savez(path, ids=np.asarray(ids), **{k: np.stack([row[k] for row in arrays]) for k in arrays[0]})
    write(out / "features.json", {
        "hook": HOOK, "tensor": path.name, "sha256": sha256(path),
        "pooling": {"q_end": "last prompt token", "r_mean": "mean excluding final-answer line and EOS"},
        "rows": {key: {"index": i, "reasoning_tokens": lengths[i], "empty_reasoning": lengths[i] == 0}
                 for i, key in enumerate(ids)},
    })
    write(out / "responses.json", responses)


def build_features(out, row, engine, seed):
    p = Problem.from_row(row)
    engine.start_cost()
    prompt = engine.prompt(p)
    if "release_bank" in row:
        source = row["release_bank"]
        if prompt != source["prompt_ids"]:
            raise ValueError(f"Released prompt tokens differ for {p.id}")
        initial = engine.record(p, prompt, source["output_ids"], "initial", 0, seed)
        initial.update(seed=None, source=CONSISTENCY_TAG, source_id=source["id"])
        initial["generation"] = {"algorithm": "released_greedy", "max_new_tokens": MAX_NEW[p.dataset]}
        raw = [initial]
    else:
        raw = engine.generate_many(p, prompt, "initial", [0], seed, False)
        raw.extend(engine.generate_many(p, prompt, "feature_sample", [0, 1, 2, 3], seed, True))
    arrays, lengths = zip(*(engine.features(p, r, confidence=True) for r in raw))
    save_features(out, arrays, [score(row, response) for response in raw], lengths)


def features_stage(stage, config, _cache, deps):
    engine = Engine.load(config)
    for row in problems(deps):
        stage.unit(row["id"], lambda out: build_features(out, row, engine, config["seed"]))


def read_features(row, stage, seed):
    path = stage.read_unit(row["id"])
    responses = read(path / "responses.json")
    released = "release_bank" in row
    ordered = validate_responses(row, [r for r in responses if r["role"] == "initial"],
                                 seed, "initial", [0], released=released)
    ordered.extend(validate_responses(row, [r for r in responses if r["role"] == "feature_sample"],
                                      seed, "feature_sample", [] if released else [0, 1, 2, 3]))
    if len(ordered) != len(responses):
        raise ValueError(f"Unexpected feature response role for {row['id']}")
    responses = ordered
    with np.load(path / "features.npz", allow_pickle=False) as data:
        lookup = {str(key): i for i, key in enumerate(data["ids"])}
        if len(lookup) != len(data["ids"]) or lookup.keys() != index(responses).keys():
            raise ValueError(f"Invalid feature IDs for {row['id']}")
        indices = [lookup[r["id"]] for r in responses]
        features = {name: data[name][indices] for name in ("r_mean", "q_end", "confidence")}
    return responses, features


def get_labels(row, stage):
    return read(stage.read_unit(row["id"]) / "distribution.json")["labels"]


def probes_stage(stage, config, _cache, deps):
    def build(out):
        examples = []
        for row in problems(deps):
            responses, features = read_features(row, deps["generate-features"], config["seed"])
            labels = get_labels(row, deps["label-distributions"])
            examples.append((row["split"], labels, responses, features))
        write(out / "probes.json", learn_probes(examples))
    stage.unit("all", build)


def build_rollouts(out, row, engine, seed, sources):
    p = Problem.from_row(row)
    engine.start_cost()
    sources = [r for r in sources if r["role"] == "initial" or
               (r["role"] == "feature_sample" and r["sample_index"] == 0)]
    if len(sources) == 1 and "release_bank" in row:
        sources.extend(engine.generate_many(p, sources[0]["prompt_ids"], "feature_sample", [0], seed, True))
        write(out / "sampled_source.json", score(row, sources[-1]))
    if len(sources) != 2 or sources[0]["prompt_ids"] != sources[1]["prompt_ids"]:
        raise ValueError(f"Missing or inconsistent prefix sources for {p.id}")
    prompt = sources[0]["prompt_ids"]
    metadata, residuals, responses, requests = [], [], [], []
    for j, source in enumerate(sources):
        output = source["output_ids"]
        positions = prefix_positions(len(output))
        if not positions:
            continue
        # Causal replay ends at the latest prefix. No future tokens are inputs.
        hidden, _ = engine.replay(prompt + output[:positions[-1]],
                                  [len(prompt) + position - 1 for position in positions])
        for position, value in zip(positions, hidden.cpu().numpy()):
            base = (j * MAX_NEW[p.dataset] + position) * 4
            ids = range(base, base + 4)
            requests.extend((i, output[:position]) for i in ids)
            prefix_id = f"{source['id']}/prefix/{position}"
            metadata.append({"id": prefix_id, "problem_id": p.id, "split": row["split"],
                             "source_response_id": source["id"], "prefix_length": position,
                             "prefix_sha256": digest(prompt + output[:position]),
                             "continuation_indices": list(ids), "attempts": 4, "hook": HOOK})
            residuals.append(value)
    for offset in range(0, len(requests), 16):
        raw = engine.generate_requests(p, prompt, "continuation", requests[offset:offset + 16], seed)
        responses.extend(score(row, response) for response in raw)
    keyed = index(validate_responses(row, responses, seed, "continuation", [i for i, _ in requests]))
    for record in metadata:
        selected = [keyed[response_id(p.id, "continuation", i)] for i in record.pop("continuation_indices")]
        record.update(response_ids=[r["id"] for r in selected], seeds=[r["seed"] for r in selected],
                      successes=sum(r["grading"]["correct"] for r in selected))
    path = out / "residuals.npz"
    matrix = np.asarray(residuals, np.float32).reshape(len(metadata), engine.model.config.hidden_size)
    np.savez(path, ids=np.asarray([r["id"] for r in metadata], dtype=str), residuals=matrix)
    tensor_hash = sha256(path)
    for i, record in enumerate(metadata):
        record["tensor"] = {"file": path.name, "index": i, "sha256": tensor_hash}
    write(out / "rollouts.json", metadata)
    write(out / "responses.json", responses)
    write(out / "cost.json", engine.cost())


def rollouts_stage(stage, config, _cache, deps):
    engine = Engine.load(config)
    for row in problems(deps):
        def build(out):
            responses, _ = read_features(row, deps["generate-features"], config["seed"])
            build_rollouts(out, row, engine, config["seed"], responses)
        stage.unit(row["id"], build)


def read_rollouts(row, path, seed):
    records = index(read(path / "rollouts.json"))
    responses = index(read(path / "responses.json"))
    expected = {r["id"] for r in records.values()}
    used = set()
    for record in records.values():
        n = record["prefix_length"]
        if (record["problem_id"] != row["id"] or record["split"] != row["split"] or n <= 0 or n % 4 or
            record["source_response_id"] not in {response_id(row["id"], role, 0) for role in ("initial", "feature_sample")}
            or record["attempts"] != 4 or record["id"] != f"{record['source_response_id']}/prefix/{n}"):
            raise ValueError(f"Invalid prefix provenance: {record['id']}")
        ids = record["response_ids"]
        if len(set(ids)) != 4 or used.intersection(ids):
            raise ValueError(f"Duplicate continuation ID for {record['id']}")
        used.update(ids)
        continuations = [responses[i] for i in ids]
        validate_responses(row, continuations, seed, "continuation", [r["sample_index"] for r in continuations])
        if ([item["seed"] for item in continuations] != record["seeds"] or
                sum(item["grading"]["correct"] for item in continuations) != record["successes"] or
                any(item["generation"]["prefix_length"] != n or len(item["output_ids"]) <= n or
                    digest(item["prompt_ids"] + item["output_ids"][:n]) != record["prefix_sha256"]
                    for item in continuations)):
            raise ValueError(f"Continuation provenance mismatch for {record['id']}")
    if used != responses.keys():
        raise ValueError(f"Unlinked continuation(s) for {row['id']}")
    with np.load(path / "residuals.npz", allow_pickle=False) as data:
        ids, residuals = data["ids"], data["residuals"]
        if len(ids) != len(set(ids)) or set(ids) != expected or len(ids) != len(residuals):
            raise ValueError(f"Invalid prefix IDs for {row['id']}")
    return residuals, np.asarray([records[k]["successes"] / records[k]["attempts"] for k in ids])


def rewards_stage(stage, config, _cache, deps):
    def build(out):
        examples = []
        for row in problems(deps):
            x, y = read_rollouts(row, deps["generate-prefix-rollouts"].read_unit(row["id"]), config["seed"])
            if len(y):
                examples.append((row, x, y, get_labels(row, deps["label-distributions"])))
        weights = {}
        metadata = {}
        for head in ("global", *CLASSES):
            pools = defaultdict(list)
            for row, x, y, labels in examples:
                if head == "global" or labels[head]:
                    pools[row["split"]].append((x, y, [row["id"]] * len(y)))
            train_data = {}
            for fold in ("fit", "tune", "calibration"):
                train_data[fold] = (tuple(np.concatenate(arrays) for arrays in zip(*pools[fold]))
                                    if pools[fold] else (np.empty((0, HIDDEN_SIZE)), np.empty(0), np.array([], dtype=str)))
            state, metadata[head] = train_reward(train_data, 1 if head == "global" else 30,
                                                 config["device"], config["seed"])
            if state is not None:
                weights[head] = state
        torch.save(weights, out / "heads.pt")
        write(out / "heads.json", metadata)
    stage.unit("all", build)
