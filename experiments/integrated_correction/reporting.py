"""Problem-paired uncertainty, audit lists, and dataset-first reports."""

from collections import Counter, defaultdict

import numpy as np

from .answers import CLASSES
from .evaluation import bundles, reward_test_points
from .learning import binary_metrics, probability_metrics, weight_by_problem
from .stages import load_bank, problems
from .storage import read, write

TYPES = ("A", *CLASSES, "B1", "C1", "E1", "E2", "unclassified")


def bootstrap(differences, seed=42):
    if not len(differences):
        return None
    rng = np.random.default_rng(seed)
    values = np.asarray(differences)
    scores = [values[rng.integers(len(values), size=len(values))].mean() for _ in range(2000)]
    return np.quantile(scores, [.025, .975]).tolist()


def conditions(rows, regime):
    if not rows:
        return {}
    all_results = {}
    for method in rows[0]["results"][regime]:
        trials = [row["results"][regime][method] for row in rows]
        initial = np.array([row["results"][regime]["greedy"]["response"]["grading"]["correct"] for row in rows])
        final = np.array([r["response"]["grading"]["correct"] for r in trials])
        recoveries = ~initial & final
        regressions = initial & ~final
        attempted = np.array([sum(event["attempted"] for event in r["logs"]) for r in trials])
        accepted = sum(event["accepted"] for r in trials for event in r["logs"])
        triggered = sum(r.get("triggered", False) for r in trials)
        costs = [r["cost"] for r in trials]
        all_results[method] = {
            "n": len(rows), "accuracy": float(final.mean()), "accuracy_ci95": bootstrap(final),
            "wrong_to_right": int(recoveries.sum()), "right_to_wrong": int(regressions.sum()),
            "net_gain": float(final.mean() - initial.mean()),
            "net_gain_ci95": bootstrap(final.astype(int) - initial),
            "intervention_attempts": int(attempted.sum()), "interventions_accepted": int(accepted),
            "recoveries_per_attempt": float((recoveries & (attempted > 0)).sum() / attempted.sum()) if attempted.sum() else None,
            "triggered": triggered, "budget_exhausted": sum(r.get("budget_exhausted", False) for r in trials),
            "global_fallback_rate": sum(r.get("weights") == {"global": 1.} for r in trials) / triggered
                if triggered and method in ("routed", "shuffled", "oracle") else None,
            "selected_head_counts": dict(Counter(len(r.get("weights", {})) for r in trials if r.get("triggered"))),
            "forward_tokens": sum(c["forward_tokens"] for c in costs),
            "generated_tokens": sum(c["generated_tokens"] for c in costs),
            "seconds": sum(c["seconds"] for c in costs),
            "peak_gpu_bytes": max(c["peak_gpu_bytes"] for c in costs),
        }
    routed = np.array([r["results"][regime]["routed"]["response"]["grading"]["correct"] for r in rows], dtype=int)
    for comparison in ("global", "generic", "shuffled", "k5_probe_vote", "k5_majority"):
        other = np.array([r["results"][regime][comparison]["response"]["grading"]["correct"] for r in rows], dtype=int)
        all_results["routed"].setdefault("paired_vs", {})[comparison] = {
            "delta": float((routed - other).mean()), "ci95": bootstrap(routed - other)}
    return all_results


def probe_metrics(rows, predictor, target):
    values = [(target(row), predictor(row)) for row in rows]
    valid = [(y, p) for y, p in values if p is not None]
    count = sum(y for y, _ in values)
    metrics = binary_metrics(*zip(*valid)) if valid else None
    return {"positive": count, "negative": len(values) - count, "scored": len(valid), "metrics": metrics}


def summarize(rows, points, audit, cap):
    def incorrect(row):
        return not row["results"]["fixed"]["greedy"]["response"]["grading"]["correct"]

    error = {"r_mean": probe_metrics(rows, lambda row: row["risk"], incorrect)}
    for view in ("q_end", "confidence", "tfidf"):
        error[view] = probe_metrics(rows, lambda row: row["baselines"][f"{view}/incorrect"], incorrect)
    classes = {}
    for label in CLASSES:
        by_view = {}
        for subset, members in (("all", rows), ("initially_wrong", [r for r in rows if incorrect(r)])):
            predictions = {"r_mean": lambda row: row["probabilities"][label],
                           "r_mean_k5": lambda row: row["k5_probabilities"]["fixed"].get(label)}
            predictions.update({name: (lambda row, name=name: row["baselines"][f"{name}/{label}"])
                                for name in ("q_end", "confidence", "tfidf")})
            by_view[subset] = {view: probe_metrics(members, predict, lambda row: row["labels"][label])
                               for view, predict in predictions.items()}
        classes[label] = by_view
    rewards = {}
    for head in ("global", *CLASSES):
        data = [(identifier, target, probabilities[head]) for identifier, target, probabilities in points
                if head in probabilities and len(target)]
        if data:
            ids = np.concatenate([[identifier] * len(y) for identifier, y, _ in data])
            metrics = probability_metrics(np.concatenate([x[1] for x in data]),
                                          np.concatenate([x[2] for x in data]), weight_by_problem(ids))
            rewards[head] = {"problems": len(data), **metrics}
        else:
            rewards[head] = None
    distribution_counts = {}
    for fold, distributions in audit.items():
        counts, unknown, failures = Counter(), Counter(), Counter()
        for dist in distributions:
            counts.update({label: int(dist[label]) for label in TYPES if dist[label] is not None})
            unknown.update(label for label in TYPES if dist[label] is None)
            failures.update(dist["failures"])
        distribution_counts[fold] = {"n": len(distributions), "types": dict(counts), "unknown_types": dict(unknown), "failures": dict(failures)}
    return {"test_problems": len(rows),
            "risk_trigger_rate": sum(r["triggered"] for r in rows) / len(rows) if rows else None,
            "probes": error, "classes": classes, "reward_test": rewards, "label_distributions": distribution_counts,
            "fixed": conditions(rows, "fixed"), "budget": conditions(rows, "budget"),
            "budget_cap_forward_tokens": cap * len(rows),
            "by_type": {label: conditions([r for r in rows if r["labels"][label]], "fixed") for label in TYPES}}


def markdown(summary, limit, release_bank=None):
    lines = ["# Integrated correction", "",
             f"Problems per dataset limit: {limit if limit is not None else 'official full splits'}.",
             "Oracle uses official test distribution labels after the primary predictions.",
             "Paired problem bootstrap uses 2,000 draws with seed 42.", ""]
    if release_bank:
        lines.extend(["GSM8K release-assisted run. Training labels use published T=1.0, top_p=1.0 answers.",
                      "Training refusals and sampled termination are unavailable. Training features use K=1.", ""])
    for dataset, result in summary.items():
        lines.extend([f"## {dataset}", "", f"Official test problems: {result['test_problems']}.",
                      f"Risk trigger rate: {result['risk_trigger_rate']}.", ""])
        for regime in ("fixed", "budget"):
            lines.extend([f"### {regime}", ""])
            if regime == "budget":
                lines.extend([f"Common cumulative cap: {result['budget_cap_forward_tokens']:,} forward tokens.", ""])
            lines.extend(["| Condition | Accuracy | W→R | R→W | Net 95% CI | Forward tokens | Generated tokens | Seconds | Peak GPU GiB |",
                          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
            for name, values in result[regime].items():
                low, high = values["net_gain_ci95"]
                lines.append(f"| {name} | {values['accuracy']:.2%} | {values['wrong_to_right']} | {values['right_to_wrong']} "
                             f"| [{low:.2%}, {high:.2%}] | {values['forward_tokens']:,} | {values['generated_tokens']:,} "
                             f"| {values['seconds']:.2f} | {values['peak_gpu_bytes'] / 1024**3:.2f} |")
            lines.append("")
    lines.extend(["See report.json for subgroup, calibration, head availability, and paired comparisons.", ""])
    return "\n".join(lines)


def add_failures(failures, dataset, responses):
    for response in responses:
        status = response["grading"]["status"]
        if status != "parsed":
            failures[response["id"]] = {"id": response["id"], "dataset": dataset,
                                          "split": response["split"], "status": status}


def for_audit(dist):
    return {**dist["labels"], **dist["subtypes"], "unclassified": dist["unclassified"], "failures": dist["failures"]}


def report_stage(stage, config, _cache, deps):
    def build(out):
        probe_bundle, heads = bundles(deps, config)
        evaluated = deps["evaluate"]
        by_dataset, points, audits = defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(list))
        failures = {}
        cluster_failures = []
        for row in problems(deps, test=True):
            key, dataset = row["id"], row["dataset"]
            main = evaluated.read_unit(f"main/{key}")
            oracle = evaluated.read_unit(f"oracle/{key}")
            reward = evaluated.read_unit(f"reward/{key}")
            result = read(main / "evaluation.json")
            dist = read(oracle / "distribution.json")
            labels = for_audit(dist)
            cluster_failures.extend(f for f in dist["failure_records"] if f["status"].startswith("cluster_"))
            result["labels"] = labels
            oracle_results = read(oracle / "oracle.json")
            for regime, methods in result["results"].items():
                methods["oracle"] = oracle_results[regime]
                add_failures(failures, dataset, [r["response"] for r in methods.values()])
            target, predictions = reward_test_points(row, reward, config["seed"], heads, labels)
            points[dataset].append((key, target, predictions))
            by_dataset[dataset].append(result)
            audits[dataset]["test"].append(labels)
            for path, name in ((main, "candidates.json"), (oracle, "responses.json"), (reward, "responses.json")):
                add_failures(failures, dataset, read(path / name))
        for row in problems(deps):
            key, dataset = row["id"], row["dataset"]
            dist = read(deps["label-distributions"].read_unit(key) / "distribution.json")
            audits[dataset][row["split"]].append(for_audit(dist))
            cluster_failures.extend(f for f in dist["failure_records"] if f["status"].startswith("cluster_"))
            add_failures(failures, dataset, load_bank(row, deps["generate-label-bank"], config["seed"]))
            for name in ("generate-features", "generate-prefix-rollouts"):
                artifact = deps[name].read_unit(key)
                add_failures(failures, dataset, read(artifact / "responses.json"))
                sampled = artifact / "sampled_source.json"
                if sampled.exists():
                    add_failures(failures, dataset, [read(sampled)])
        datasets = ("gsm8k",) if config.get("gsm8k_release_bank") else ("gsm8k", "math")
        cap = config["forward_budget_per_problem"]
        summary = {name: summarize(by_dataset[name], points[name], audits[name], cap) for name in datasets}
        combined_audit = {fold: [entry for name in datasets for entry in audits[name][fold]]
                          for fold in ("fit", "tune", "calibration", "test")}
        summary["combined"] = summarize(sum((by_dataset[name] for name in datasets), []),
                                        sum((points[name] for name in datasets), []), combined_audit, cap)
        metadata = read(deps["train-rewards"].read_unit("all") / "heads.json")
        provenance = read(deps["prepare"].read_unit("all") / "provenance.json")
        write(out / "report.json", {"results": summary, "seed": config["seed"], "bootstrap_seed": 42,
                                    "budget_per_problem": cap, "bootstrap_draws": 2000,
                                    "lock": read(deps["tune-correction"].read_unit("locked") / "locked.json"),
                                    "probes": {name: {k: v for k, v in probe.items()
                                                      if k in ("disabled", "fit", "calibration", "c", "tune_auroc")}
                                               for name, probe in probe_bundle["probes"].items()},
                                    "reward_heads": metadata, "gsm8k_release_bank": provenance.get("gsm8k_release_bank")})
        write(out / "failures.json", {"responses": list(failures.values()), "clustering": cluster_failures,
                                      "missing_gold": provenance["missing_gold"],
                                      "removed_train_test_duplicates": provenance["removed_train_test_duplicates"]})
        (out / "report.md").write_text(markdown(summary, config["limit_per_dataset"], provenance.get("gsm8k_release_bank")), encoding="utf-8")
    stage.unit("all", build)
