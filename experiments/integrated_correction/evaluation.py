"""Tune-time model selection and locked, problem-matched test comparisons."""

from collections import defaultdict
from dataclasses import asdict, replace
from itertools import product

import numpy as np

from .answers import CLASSES, distribution
from .correction import (ALPHAS, REWARD_THRESHOLDS, Assessment, CorrectionConfig, assess,
                         class_probabilities, correct, regenerate, route, vote)
from .learning import load_rewards, predict_probe, score_reward
from .model import BudgetExhausted, Engine, Problem
from .stages import build_bank, build_rollouts, problems, read_features, read_rollouts, score
from .storage import read, write


def bundles(deps, config):
    probes = read(deps["train-probes"].read_unit("all") / "probes.json")
    rewards = deps["train-rewards"].read_unit("all")
    heads = load_rewards(rewards / "heads.pt", read(rewards / "heads.json"), config["device"])
    return probes, heads


def tune_stage(stage, config, _cache, deps):
    probes, heads = bundles(deps, config)
    candidates = list(product(ALPHAS, REWARD_THRESHOLDS))
    rows = [r for r in problems(deps) if r["split"] == "tune"]
    engine = Engine.load(config) if rows else None
    totals = [{"alpha": a, "threshold": t, "correct": 0, "forward_tokens": 0} for a, t in candidates]
    for row in rows:
        def build(out):
            p = Problem.from_row(row)
            responses, f = read_features(row, deps["generate-features"], config["seed"])
            initial = {k: v for k, v in responses[0].items() if k not in ("grading", "split")}
            assessment = assess(initial, f["r_mean"][0], probes)
            outcomes = []
            for i, (alpha, threshold) in enumerate(candidates):
                engine.start_cost()
                settings = CorrectionConfig(engine, config["seed"], alpha, threshold, i)
                response, logs = correct(p, assessment, heads, settings)
                outcomes.append({"graded_response": score(row, response), "logs": logs, "cost": engine.cost()})
            write(out / "tuning.json", outcomes)
        directory = stage.unit(row["id"], build)
        for total, trial in zip(totals, read(directory / "tuning.json")):
            total["correct"] += trial["graded_response"]["grading"]["correct"]
            total["forward_tokens"] += trial["cost"]["forward_tokens"]
    best = min(totals, key=lambda t: (-t["correct"], t["forward_tokens"], t["alpha"], t["threshold"]))
    stage.unit("locked", lambda out: write(out / "locked.json", {
        "alpha": best["alpha"], "threshold": best["threshold"], "tune_problems": len(rows),
        "status": "tuned" if rows else "no_tune_rows_grid_first", "settings_grid": totals,
        "risk_threshold": probes["risk_threshold"], "global_available": "global" in heads,
        "threshold_calibration_n": probes["threshold_calibration_n"],
        "threshold_calibration_rate": probes["threshold_calibration_rate"],
    }))


def add_cost(*values):
    costs = {name: sum(v[name] for v in values) for name in ("forward_tokens", "generated_tokens", "seconds")}
    return {**costs, "peak_gpu_bytes": max((v["peak_gpu_bytes"] for v in values), default=0)}


def run_method(p, assessment, heads, engine, seed, lock, key, index, baseline_cost, cap):
    engine.start_cost(None if cap is None else max(0, cap - baseline_cost["forward_tokens"]))
    logs, exhausted = [], False
    weights = ({"global": 1.} if "global" in heads else {}) if key == "global" else route(assessment.probabilities, heads)
    response = assessment.initial
    try:
        if assessment.triggered:
            settings = CorrectionConfig(engine, seed, lock["alpha"], lock["threshold"], index)
            if key == "generic":
                response, = engine.generate_many(p, assessment.initial["prompt_ids"], "corrected", [index], seed, True)
            elif weights and key == "routed":
                response, logs = correct(p, assessment, heads, settings)
            elif weights:
                response, logs = regenerate(p, assessment, heads, settings, weights)
    except BudgetExhausted as error:
        exhausted = True
        logs = getattr(error, "logs", [])
    return {"response": response, "logs": logs, "cost": add_cost(baseline_cost, engine.cost()),
            "triggered": assessment.triggered, "budget_exhausted": exhausted,
            "weights": weights if key != "generic" and assessment.triggered else {},
            "probabilities": assessment.probabilities}


def run_voting(p, initial, initial_vector, probes, engine, seed, initial_cost, feature_cost, cap):
    engine.start_cost(None if cap is None else max(0, cap - initial_cost["forward_tokens"]))
    candidates = [initial]
    majority = initial
    voted = initial
    exhausted = False
    probabilities = {}
    try:
        candidates += engine.generate_many(p, initial["prompt_ids"], "feature_sample", [0, 1, 2, 3], seed, True)
        majority = vote(p, candidates, [1.] * len(candidates))
    except BudgetExhausted:
        exhausted = True
    generation_cost = add_cost(initial_cost, engine.cost())
    majority_result = {"response": majority, "logs": [], "cost": generation_cost, "budget_exhausted": exhausted}
    if not exhausted:
        baseline = add_cost(generation_cost, feature_cost)
        engine.start_cost(None if cap is None else max(0, cap - baseline["forward_tokens"]))
        try:
            if cap is not None and baseline["forward_tokens"] > cap:
                raise BudgetExhausted
            vectors = np.stack([initial_vector, *(engine.features(p, r)[0]["r_mean"] for r in candidates[1:])])
            scores = [predict_probe(probes["probes"]["r_mean/incorrect"], f) for f in vectors]
            voted = vote(p, candidates, [1. - (score or 0.) for score in scores])
            probabilities = class_probabilities(probes["probes"], "r_mean_k5", vectors.mean(0))
        except BudgetExhausted:
            exhausted = True
        if cap is None or baseline["forward_tokens"] <= cap:
            generation_cost = add_cost(baseline, engine.cost())
    return {"k5_majority": majority_result,
            "k5_probe_vote": {"response": voted, "logs": [], "cost": generation_cost, "budget_exhausted": exhausted}
            }, candidates, probabilities


def initial_test(out, p, engine, probes, seed):
    engine.start_cost()
    initial, = engine.generate_many(p, engine.prompt(p), "initial", [0], seed, False)
    initial_cost = engine.cost()
    engine.start_cost()
    features, length = engine.features(p, initial)
    assessment = assess(initial, features["r_mean"], probes)
    feature_cost = engine.cost()
    features["confidence"] = engine.features(p, initial, confidence=True)[0]["confidence"]
    baseline = {f"{view}/{target}": predict_probe(
        probes["probes"][f"{view}/{target}"], initial["text"] if view == "tfidf" else features[view])
        for view in ("q_end", "confidence", "tfidf") for target in ("incorrect", *CLASSES)}
    write(out / "initial.json", {"assessment": asdict(assessment), "initial_cost": initial_cost,
                                 "feature_cost": feature_cost, "baseline": baseline, "reasoning_tokens": length})
    np.save(out / "r_mean.npy", features["r_mean"], allow_pickle=False)


def permutations(rows, stats, seed):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row["id"])
    shuffled = {}
    rng = np.random.default_rng(seed)
    for ids in grouped.values():
        ids = sorted(ids)
        for target, source in zip(ids, rng.permutation(ids)):
            shuffled[target] = {"source_problem": str(source), "probabilities": stats[source]["assessment"]["probabilities"]}
    return shuffled


def eval_stage(stage, config, _cache, deps):
    lock = read(deps["tune-correction"].read_unit("locked") / "locked.json")
    probes, heads = bundles(deps, config)
    if lock["risk_threshold"] != probes["risk_threshold"]:
        raise ValueError("Test evaluation requires the locked calibration threshold")
    rows = problems(deps, test=True)
    engine = Engine.load(config)
    seed, budget = config["seed"], config["forward_budget_per_problem"]
    stats = {}
    for row in rows:
        p = Problem.from_row(row)
        directory = stage.unit(f"initial/{p.id}", lambda out: initial_test(out, p, engine, probes, seed))
        stats[p.id] = read(directory / "initial.json")
    shuffled = permutations(rows, stats, seed)

    for row in rows:
        p = Problem.from_row(row)
        item = stats[p.id]
        assessment = Assessment(**item["assessment"])
        baseline_cost = add_cost(item["initial_cost"], item["feature_cost"])

        def build_main(out):
            initial_dir = stage.read_unit(f"initial/{p.id}")
            vector = np.load(initial_dir / "r_mean.npy", allow_pickle=False)
            results, k5 = {}, {}
            sampled = []
            for regime, cap in (("fixed", None), ("budget", budget)):
                baseline = {"response": assessment.initial, "logs": [], "cost": item["initial_cost"]}
                if cap is not None and item["initial_cost"]["forward_tokens"] > cap:
                    raise ValueError(f"Budget is too small for the common initial response: {p.id}")
                methods = {"greedy": baseline}
                for i, name in enumerate(("generic", "global", "routed", "shuffled")):
                    prediction = (replace(assessment, probabilities=shuffled[p.id]["probabilities"])
                                  if name == "shuffled" else assessment)
                    if cap is not None and baseline_cost["forward_tokens"] > cap:
                        methods[name] = {**baseline, "budget_exhausted": True}
                        continue
                    methods[name] = run_method(p, prediction, heads, engine, seed, lock, name,
                                               i, baseline_cost, cap)
                voting, candidates, predictions = run_voting(
                    p, assessment.initial, vector, probes, engine, seed, item["initial_cost"], item["feature_cost"], cap)
                methods.update(voting)
                results[regime] = {name: {**data, "response": score(row, data["response"])}
                                   for name, data in methods.items()}
                k5[regime] = predictions
                if cap is None:
                    sampled = [score(row, r) for r in candidates]
            write(out / "evaluation.json", {"id": p.id, "dataset": p.dataset, "results": results,
                                            "risk": assessment.risk, "triggered": assessment.triggered,
                                            "probabilities": assessment.probabilities, "k5_probabilities": k5,
                                            "shuffle": shuffled[p.id], "baselines": item["baseline"]})
            write(out / "candidates.json", sampled)

        main = stage.unit(f"main/{p.id}", build_main)

        def build_oracle(out):
            sampling = {"temperature": 1., "top_p": 1.} if config.get("gsm8k_release_bank") else {}
            bank = build_bank(row, engine, seed, **sampling)
            labels = distribution(p.dataset, bank)
            diagnostic = replace(assessment, probabilities={k: float(labels["labels"][k]) for k in CLASSES})
            results = {}
            for regime, cap in (("fixed", None), ("budget", budget)):
                if cap is not None and baseline_cost["forward_tokens"] > cap:
                    result = {"response": assessment.initial, "logs": [], "cost": item["initial_cost"],
                              "budget_exhausted": True}
                else:
                    result = run_method(p, diagnostic, heads, engine, seed, lock, "oracle",
                                        4, baseline_cost, cap)
                results[regime] = {**result, "response": score(row, result["response"])}
            write(out / "responses.json", bank)
            write(out / "distribution.json", labels)
            write(out / "oracle.json", results)

        stage.unit(f"oracle/{p.id}", build_oracle)
        stage.unit(f"reward/{p.id}", lambda out: build_rollouts(
            out, row, engine, seed, read(main / "candidates.json")))


def reward_test_points(row, path, seed, heads, labels):
    import torch

    x, y = read_rollouts(row, path, seed)
    predictions = {}
    for name, (head, cal) in heads.items():
        if name == "global" or labels[name]:
            logits = torch.from_numpy(score_reward(head, x))
            predictions[name] = torch.sigmoid(logits * cal["slope"] + cal["intercept"]).numpy()
    return y, predictions
