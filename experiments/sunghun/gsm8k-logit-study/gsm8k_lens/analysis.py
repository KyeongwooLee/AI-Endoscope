from pathlib import Path
import math

import numpy as np
import pandas as pd

from .config import manifest_for_run
from .pipeline import rows_for_phase, stage_fingerprint, require_gate
from .storage import (commit_files, complete, file_digest, read_json, source_digest, write_json,
                      write_parquet, write_text, atomic_write)


PLOT_FEATURES = ["entropy", "margin", "generated_logprob", "log_rank", "js_final",
                 "top1_agreement", "adjacent_js", "support_change", "lower_tail_logprob"]
COLORS = {False: "#bd4653", True: "#187c79"}


def bootstrap_mean(values, repetitions, seed):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return (float("nan"),) * 3
    means = bootstrap_samples(values, repetitions, np.random.default_rng(seed))
    return float(values.mean()), *map(float, np.quantile(means, [0.025, 0.975]))


def bootstrap_samples(values, repetitions, rng):
    result = []
    for start in range(0, repetitions, 100):
        n = min(100, repetitions - start)
        indices = rng.integers(0, len(values), size=(n, len(values)))
        result.extend(values[indices].mean(axis=1).tolist())
    return np.asarray(result)


def difference_ci(correct, incorrect, repetitions, seed):
    correct, incorrect = (np.asarray(x, dtype=float) for x in (correct, incorrect))
    correct, incorrect = correct[np.isfinite(correct)], incorrect[np.isfinite(incorrect)]
    if not len(correct) or not len(incorrect):
        return {"difference": None, "low": None, "high": None}
    rng = np.random.default_rng(seed)
    boot = bootstrap_samples(incorrect, repetitions, rng) - bootstrap_samples(correct, repetitions, rng)
    low, high = np.quantile(boot, [0.025, 0.975])
    return {"difference": float(incorrect.mean() - correct.mean()), "low": float(low), "high": float(high)}


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def analyze(cfg, run, phase, allow_partial=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Path(run)
    manifest = manifest_for_run(run, cfg)
    require_gate(run, manifest, phase)
    rows = rows_for_phase(run, manifest, phase)
    fingerprint = stage_fingerprint(manifest, "extract")
    generation_fp = stage_fingerprint(manifest, "generate")
    outcomes, examples, trajectories, completed_ids = [], [], [], []
    for row in rows:
        sample_id = row["id"]
        gen_dir = run / "generations" / phase / sample_id
        label_path = run / "labels" / phase / f"{sample_id}.json"
        if not complete(gen_dir, generation_fp, verify=True) or not label_path.exists():
            if not allow_partial:
                raise ValueError(f"Missing generation/label: {sample_id}; use --allow-partial for pilot inspection")
            continue
        generation, label = read_json(gen_dir / "generation.json"), read_json(label_path)
        if label["generation_checksum"] != file_digest(gen_dir / "generation.json"):
            raise ValueError(f"Stale label: {sample_id}")
        if label.get("parser_hash") != source_digest(["answers.py"]):
            raise ValueError("Parser changed; run label again")
        outcome = {**label, "generated_tokens": generation["generated_tokens"]}
        outcomes.append(outcome)
        folder = run / "features" / phase / sample_id
        if not complete(folder, fingerprint, verify=True):
            if not allow_partial:
                raise ValueError(f"Missing features: {sample_id}")
            continue
        if not read_json(folder / "quality.json")["passed"]:
            raise ValueError(f"Failed numerical check: {sample_id}")
        completed_ids.append(sample_id)
        for filename, target in (("example_features.parquet", examples), ("trajectory.parquet", trajectories)):
            frame = pd.read_parquet(folder / filename)
            frame["answer_correct"] = label["answer_correct"]
            frame["analysis_eligible"] = label["analysis_eligible"]
            frame["generated_tokens"] = generation["generated_tokens"]
            target.append(frame)
    if not outcomes or not examples:
        raise ValueError("No complete analysis inputs")
    outcome_frame = pd.DataFrame(outcomes)
    example_frame = pd.concat(examples, ignore_index=True)
    trajectory_frame = pd.concat(trajectories, ignore_index=True)
    target = run / "analysis" / phase
    target.mkdir(parents=True, exist_ok=True)
    (target / "complete.json").unlink(missing_ok=True)
    figures = target / "figures"
    figures.mkdir(exist_ok=True)
    write_parquet(target / "example_features.parquet", example_frame)
    write_parquet(target / "trajectory.parquet", trajectory_frame)
    valid = example_frame[example_frame.analysis_eligible].copy()
    ana = cfg["analysis"]
    primary = valid[(valid.layer == ana["primary_layer"]) & (valid.region == "reasoning")].copy()
    correct = primary.loc[primary.answer_correct, "js_final"].to_numpy()
    incorrect = primary.loc[~primary.answer_correct, "js_final"].to_numpy()
    result = difference_ci(correct, incorrect, ana["bootstrap_replicates"], cfg["seed"])
    summary = {"phase": phase, "expected": len(rows), "generated_and_labeled": len(outcomes),
               "completed": len(completed_ids), "partial": len(completed_ids) != len(rows),
               "strict_accuracy": float(outcome_frame.answer_correct.mean()),
               "parse_failure_rate": float((outcome_frame.parse_status != "ok").mean()),
               "truncation_rate": float(outcome_frame.truncated.mean()),
               "eligible_generation_coverage": float(outcome_frame.analysis_eligible.mean()),
               "primary_coverage": len(primary) / len(outcomes),
               "eligible_without_reasoning": int(outcome_frame[
                   outcome_frame.id.isin(completed_ids)].analysis_eligible.sum()) - len(primary),
               "n_correct": len(correct), "n_incorrect": len(incorrect),
               "small_group": min(len(correct), len(incorrect)) < ana["minimum_group_size"],
               "primary_layer": ana["primary_layer"], "primary_feature": "reasoning_mean_js_final",
               "primary": result, "source_hash": source_digest(), "config_hash": manifest["config_hash"]}
    if result["low"] is None:
        interpretation = "Not estimable: one outcome group has no eligible reasoning examples."
    elif result["low"] <= 0 <= result["high"]:
        interpretation = "The 95% interval includes zero; a group difference is not established."
    else:
        interpretation = "A group difference is observed under this protocol; it does not establish individual error detection or causality."
    summary["interpretation"] = interpretation
    write_json(target / "summary.json", clean_json(summary))
    flat_summary = {**{k: v for k, v in summary.items() if k != "primary"}, **result}
    atomic_write(target / "summary.csv", lambda p: pd.DataFrame([flat_summary]).to_csv(p, index=False))

    estimates = []
    for (region, layer, outcome), group in valid.groupby(["region", "layer", "answer_correct"]):
        for feature in PLOT_FEATURES:
            mean, low, high = bootstrap_mean(group[feature], ana["bootstrap_replicates"], cfg["seed"])
            estimates.append({"region": region, "layer": layer, "answer_correct": outcome,
                              "feature": feature, "mean": mean, "low": low, "high": high,
                              "n": int(group[feature].notna().sum())})
    stats = pd.DataFrame(estimates, columns=["region", "layer", "answer_correct", "feature", "mean", "low", "high", "n"])
    atomic_write(target / "group_statistics.csv", lambda p: stats.to_csv(p, index=False))
    for feature in PLOT_FEATURES:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
        for ax, region in zip(axes, ("reasoning", "final_number", "final_first_token")):
            for outcome in (True, False):
                data = stats[(stats.feature == feature) & (stats.region == region)
                             & (stats.answer_correct == outcome)].sort_values("layer")
                if len(data):
                    ax.plot(data.layer, data["mean"], color=COLORS[outcome], label="Correct" if outcome else "Incorrect")
                    ax.fill_between(data.layer, data.low, data.high, color=COLORS[outcome], alpha=0.17)
            ax.set(title=region, xlabel="Layer", ylabel=feature)
            ax.grid(alpha=0.15)
            if ax.lines:
                ax.legend()
        fig.tight_layout()
        fig.savefig(figures / f"layers_{feature}.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for outcome, values in ((True, correct), (False, incorrect)):
        if len(values):
            sorted_values = np.sort(values)
            ax.step(sorted_values, np.arange(1, len(values) + 1) / len(values), where="post",
                    label=f"{'Correct' if outcome else 'Incorrect'} (n={len(values)})", color=COLORS[outcome])
    ax.set(xlabel=f"Mean reasoning JS: layer {ana['primary_layer']} vs final", ylabel="ECDF")
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "primary_distribution.png", dpi=150)
    plt.close(fig)

    eligible_trajectory = trajectory_frame[trajectory_frame.analysis_eligible]
    group_maps, count_maps = [], []
    for outcome in (True, False):
        subset = eligible_trajectory[eligible_trajectory.answer_correct == outcome]
        grouped = subset.groupby(["layer", "bin"]).js_final.agg(["mean", "count"]).reset_index()
        atomic_write(target / f"trajectory_{'correct' if outcome else 'incorrect'}.csv",
                     lambda p, g=grouped: g.to_csv(p, index=False))
        grid = grouped.pivot(index="layer", columns="bin", values="mean").reindex(
            index=range(1, cfg["model"]["expected_layers"] + 1), columns=range(ana["trajectory_bins"]))
        group_maps.append(grid.to_numpy(dtype=float))
        counts = grouped.pivot(index="layer", columns="bin", values="count").reindex(
            index=range(1, cfg["model"]["expected_layers"] + 1), columns=range(ana["trajectory_bins"]))
        count_maps.append(counts.fillna(0).to_numpy(dtype=float))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    maps = [*group_maps, group_maps[1] - group_maps[0]]
    finite = np.concatenate([g[np.isfinite(g)] for g in group_maps])
    vmax = float(finite.max()) if len(finite) else 1.0
    for i, (ax, grid, name) in enumerate(zip(axes, maps, ("Correct", "Incorrect", "Incorrect - Correct"))):
        if i == 2:
            finite_diff = np.abs(grid[np.isfinite(grid)])
            bound = max(float(finite_diff.max()) if len(finite_diff) else 0, 1e-6)
            im = ax.imshow(grid, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound)
        else:
            im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=max(vmax, 1e-6))
        ax.set(title=name, xlabel="Relative reasoning bin", ylabel="Layer (index + 1)")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(figures / "trajectory_js_final.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, grid, name in zip(axes, count_maps, ("Correct", "Incorrect")):
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="cividis", vmin=0,
                       vmax=max(1, max(g.max() for g in count_maps)))
        ax.set(title=f"{name}: contributing problems", xlabel="Relative reasoning bin", ylabel="Layer (index + 1)")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(figures / "trajectory_counts.png", dpi=150)
    plt.close(fig)

    depth = valid[(valid.layer == 1) & (valid.region == "reasoning")]
    fig, ax = plt.subplots(figsize=(7, 4))
    for outcome in (True, False):
        values = depth.loc[depth.answer_correct == outcome, "stabilization_depth"].dropna().sort_values()
        if len(values):
            ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post",
                    color=COLORS[outcome], label="Correct" if outcome else "Incorrect")
    ax.set(xlabel="Per-problem mean prediction stabilization depth", ylabel="ECDF")
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "stabilization_depth.png", dpi=150)
    plt.close(fig)

    lengths = outcome_frame[outcome_frame.analysis_eligible].generated_tokens.to_numpy()
    if phase == "test":
        boundaries = read_json(run / "analysis/exploration/length_bins.json")["boundaries"]
    else:
        boundaries = np.quantile(lengths, [0.25, 0.5, 0.75]).tolist() if len(lengths) else []
        write_json(target / "length_bins.json", {"boundaries": boundaries, "source_phase": phase})
    length_rows = []
    if boundaries:
        primary["length_bin"] = np.searchsorted(boundaries, primary.generated_tokens, side="right")
        for bin_id, group in primary.groupby("length_bin"):
            for outcome in (True, False):
                values = group.loc[group.answer_correct == outcome, "js_final"]
                mean, low, high = bootstrap_mean(values, ana["bootstrap_replicates"], cfg["seed"])
                length_rows.append({"bin": int(bin_id), "answer_correct": outcome, "n": len(values),
                                    "mean": mean, "low": low, "high": high})
    length_table = pd.DataFrame(length_rows, columns=["bin", "answer_correct", "n", "mean", "low", "high"])
    atomic_write(target / "length_comparison.csv", lambda p: length_table.to_csv(p, index=False))
    fig, ax = plt.subplots(figsize=(7, 4))
    for outcome in (True, False):
        data = length_table[(length_table.answer_correct == outcome) & (length_table.n > 0)]
        if len(data):
            ax.plot(data.bin, data["mean"], "o-", color=COLORS[outcome], label="Correct" if outcome else "Incorrect")
            ax.fill_between(data.bin, data.low, data.high, color=COLORS[outcome], alpha=0.17)
    ax.set(xlabel="Generation length quartile (0-3)", ylabel="Primary JS feature")
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "length_control.png", dpi=150)
    plt.close(fig)

    final = valid[(valid.layer == cfg["model"]["expected_layers"]) & (valid.region == "reasoning")]
    joined = primary[["id", "js_final"]].merge(final[["id", "entropy", "margin", "generated_logprob"]], on="id")
    atomic_write(target / "confidence_correlations.csv", lambda p: joined.drop(columns="id").corr().to_csv(p))
    example_text = []
    rng = np.random.default_rng(cfg["seed"])
    for outcome in (True, False):
        ids = sorted(primary.loc[primary.answer_correct == outcome, "id"].tolist())
        selected = rng.choice(ids, size=min(3, len(ids)), replace=False).tolist() if ids else []
        for sample_id in selected:
            generation = read_json(run / "generations" / phase / sample_id / "generation.json")
            tokens = pd.read_parquet(run / "features" / phase / sample_id / "token_features.parquet")
            grid = tokens.pivot(index="layer", columns="token_index", values="js_final")
            fig, ax = plt.subplots(figsize=(11, 4))
            im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
            ax.set(title=f"{sample_id}: {'correct' if outcome else 'incorrect'}",
                   xlabel="Generated target token index", ylabel="Layer (index + 1)")
            fig.colorbar(im, ax=ax, label="JS to final")
            fig.tight_layout()
            fig.savefig(figures / f"example_{sample_id}.png", dpi=150)
            plt.close(fig)
            example_text.append(f"### {sample_id}: {'correct' if outcome else 'incorrect'}\n\n"
                                f"```text\n{generation['text']}\n```\n\n"
                                f"![Token trajectory](figures/example_{sample_id}.png)\n")
    report = (
        f"# {phase} logit-lens report\n\n"
        f"Status: {'PARTIAL / diagnostic only' if summary['partial'] else 'complete'}\n\n"
        f"Generated/labeled: {len(outcomes)}/{len(rows)}. Features complete: {len(completed_ids)}.\n\n"
        f"Strict final-answer accuracy: {summary['strict_accuracy']:.4f}. "
        f"Parse failures: {summary['parse_failure_rate']:.4f}. Truncation: {summary['truncation_rate']:.4f}.\n\n"
        f"Primary analysis: {len(correct)} correct, {len(incorrect)} incorrect; "
        f"coverage {summary['primary_coverage']:.4f}. Small group: {summary['small_group']}.\n\n"
        f"Incorrect minus correct mean JS: {result['difference']}; 95% CI [{result['low']}, {result['high']}].\n\n"
        f"{interpretation}\n\n"
        "Layer/feature searches are exploratory. No first-error-step labels or causal claims are available. "
        "Length stratification does not control question difficulty or logit-lens calibration.\n\n"
        "![Primary distribution](figures/primary_distribution.png)\n\n"
        "![Trajectory](figures/trajectory_js_final.png)\n\n"
        "![Trajectory counts](figures/trajectory_counts.png)\n\n"
        "![Stabilization](figures/stabilization_depth.png)\n\n"
        "![Length control](figures/length_control.png)\n\n"
        + "\n".join(example_text)
    )
    write_text(target / "report.md", report)
    commit_files(target, [str(p.relative_to(target)).replace("\\", "/")
                          for p in sorted(target.rglob("*")) if p.is_file()],
                 stage_fingerprint(manifest, "analysis"))
    print(f"Analysis saved at {target / 'report.md'}", flush=True)
    return summary
