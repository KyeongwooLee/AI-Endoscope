import math
import numpy as np
import pandas as pd
import torch


FEATURES = ["entropy", "margin", "generated_logprob", "generated_rank",
            "js_final", "top1_agreement", "adjacent_js", "support_change"]


def js_divergence(logp, logq):
    logm = torch.logaddexp(logp, logq) - math.log(2)
    return (0.5 * ((logp.exp() * (logp - logm)).sum(-1)
                   + (logq.exp() * (logq - logm)).sum(-1))).clamp_min(0)


def stable_topk(logits, k):
    # max returns the first index on ties. Keep all operations batched on GPU.
    remaining = logits.clone()
    values, indices = [], []
    for _ in range(k):
        value, index = remaining.max(dim=-1)
        values.append(value)
        indices.append(index)
        remaining.scatter_(1, index[:, None], -float("inf"))
    return torch.stack(values, dim=-1), torch.stack(indices, dim=-1)


def distribution_features(logits, target_ids, final_logp):
    logits = logits.float()
    logp = logits.log_softmax(-1)
    probability = logp.exp()
    top_values, top_ids = stable_topk(logits, min(5, logits.shape[-1]))
    chosen_logits = logits.gather(-1, target_ids[:, None]).squeeze(-1)
    chosen_logp = logp.gather(-1, target_ids[:, None]).squeeze(-1)
    top_probability = (top_values - logits.logsumexp(-1, keepdim=True)).exp()
    metrics = {
        "entropy": -(probability * logp).sum(-1),
        "margin": top_probability[:, 0] - top_probability[:, 1],
        "generated_logprob": chosen_logp,
        "generated_rank": 1 + (logits > chosen_logits[:, None]).sum(-1),
        "js_final": js_divergence(logp, final_logp),
        "top1_agreement": (top_ids[:, 0] == final_logp.argmax(-1)).float(),
    }
    return metrics, logp, top_values, top_ids, chosen_logits


def stabilization_depth(top1, start_layer=1):
    # top1 is [layers, tokens]; the final layer always agrees with itself.
    layers, _ = top1.shape
    if not 1 <= start_layer <= layers:
        raise ValueError("Invalid stabilization start layer")
    mismatch = top1[start_layer - 1:] != top1[-1:]
    layer_ids = np.arange(start_layer, layers + 1)[:, None]
    last_mismatch = np.where(mismatch, layer_ids, start_layer - 1).max(axis=0)
    return last_mismatch + 1


def lower_tail(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    return float(np.sort(values)[:max(1, math.ceil(0.1 * len(values)))].mean())


@torch.inference_mode()
def project_features(model, residual, native_hidden, target_ids, cfg):
    layers, tokens, _ = residual.shape
    chunk_size = cfg["extraction"]["projection_chunk_tokens"]
    arrays = {name: np.full((layers, tokens), np.nan, dtype=np.float32) for name in FEATURES}
    top1 = np.empty((layers, tokens), dtype=np.int64)
    top_rows = []
    native_errors, generation_logp, native_top1 = [], [], []
    for start in range(0, tokens, chunk_size):
        stop = min(start + chunk_size, tokens)
        targets = torch.tensor(target_ids[start:stop], device=model.device)
        final_raw = residual[-1, start:stop].to(model.device)
        final_logits = model.lm_head(model.model.norm(final_raw)).float()
        final_logp = final_logits.log_softmax(-1)
        if native_hidden is not None:
            native_logits = model.lm_head(native_hidden[start:stop].to(model.device)).float()
            native_logp = native_logits.log_softmax(-1)
            native_errors.extend((final_logp - native_logp).abs().amax(-1).cpu().tolist())
            del native_logits, native_logp
        generation_logp.extend(final_logp.gather(-1, targets[:, None]).squeeze(-1).cpu().tolist())
        native_top1.extend(final_logp.argmax(-1).cpu().tolist())
        previous = None
        for layer in range(layers):
            if layer == layers - 1:
                logits = final_logits
            else:
                hidden = residual[layer, start:stop].to(model.device)
                logits = model.lm_head(model.model.norm(hidden)).float()
            metrics, logp, values, ids, selected_logits = distribution_features(logits, targets, final_logp)
            for name, value in metrics.items():
                arrays[name][layer, start:stop] = value.cpu().numpy()
            top1[layer, start:stop] = ids[:, 0].cpu().numpy()
            if previous is not None:
                arrays["adjacent_js"][layer - 1, start:stop] = js_divergence(previous, logp).cpu().numpy()
                arrays["support_change"][layer - 1, start:stop] = (
                    arrays["generated_logprob"][layer, start:stop]
                    - arrays["generated_logprob"][layer - 1, start:stop])
            cpu_ids, cpu_values, chosen = ids.cpu().tolist(), values.cpu().tolist(), selected_logits.cpu().tolist()
            for offset in range(stop - start):
                top_rows.append({"layer": layer + 1, "token_index": start + offset,
                                 "top5_ids": cpu_ids[offset], "top5_logits": cpu_values[offset],
                                 "generated_logit": chosen[offset]})
            previous = logp
            del metrics, values, ids, logits
        del previous, logp, final_logp, final_logits
    depth = stabilization_depth(top1)
    late_depth = stabilization_depth(top1, cfg["analysis"]["late_start_layer"])
    diagnostics = pd.DataFrame({"token_index": np.arange(tokens), "stabilization_depth": depth,
                                "late_stabilization_depth": late_depth,
                                "top1_switches": (top1[1:] != top1[:-1]).sum(axis=0)})
    rows = {"layer": np.repeat(np.arange(1, layers + 1), tokens),
            "token_index": np.tile(np.arange(tokens), layers)}
    rows.update({key: value.reshape(-1) for key, value in arrays.items()})
    quality = {"native_lens_max_logprob_error": max(native_errors) if native_errors else None,
               "replay_generated_logprob": generation_logp, "replay_top1": native_top1}
    return pd.DataFrame(rows), pd.DataFrame(top_rows), diagnostics, quality


def check_quality(quality, generation, cfg):
    tolerance = cfg["quality"]
    error = quality["native_lens_max_logprob_error"]
    quality["native_lens_pass"] = error is not None and error <= tolerance["native_lens_max_logprob_error"]
    quality["generation_replay_pass"] = None
    if "generation_quality" in generation:
        reference = generation["generation_quality"]
        errors = np.abs(np.asarray(reference["generated_logprob"])
                        - np.asarray(quality["replay_generated_logprob"]))
        agreement = float(np.mean(np.asarray(reference["top1"]) == np.asarray(quality["replay_top1"])))
        quality.update(generation_replay_max_logprob_error=float(errors.max()),
                       generation_replay_mean_logprob_error=float(errors.mean()),
                       generation_replay_top1_agreement=agreement)
        quality["generation_replay_pass"] = bool(
            errors.max() <= tolerance["generation_replay_max_logprob_error"]
            and agreement >= tolerance["generation_replay_min_top1_agreement"])
    quality["passed"] = quality["native_lens_pass"] and quality["generation_replay_pass"] is not False
    return quality


def summarize_example(tokens, diagnostics, bins):
    frame = tokens[tokens.included].copy()
    first_number = frame[frame.region == "final_number"].sort_values("token_index").groupby("layer").head(1).copy()
    first_number["region"] = "final_first_token"
    frame = pd.concat([frame, first_number], ignore_index=True)
    frame["log_rank"] = np.log1p(frame.generated_rank)
    frame["support_change_abs"] = frame.support_change.abs()
    summary = []
    for (layer, region), group in frame.groupby(["layer", "region"]):
        row = {"layer": int(layer), "region": region, "n_tokens": len(group)}
        for feature in FEATURES + ["log_rank", "support_change_abs"]:
            values = group[feature].dropna()
            row[feature] = float(values.mean()) if len(values) else float("nan")
            row[feature + "_median"] = float(values.median()) if len(values) else float("nan")
        row["lower_tail_logprob"] = lower_tail(group.generated_logprob)
        summary.append(row)
    columns = ["layer", "region", "n_tokens"] + [
        name for feature in FEATURES + ["log_rank", "support_change_abs"]
        for name in (feature, feature + "_median")] + ["lower_tail_logprob"]
    examples = pd.DataFrame(summary, columns=columns)
    diagnostic_region = tokens[tokens.layer == 1][["token_index", "region", "included"]]
    diagnostic_region = pd.concat([diagnostic_region, first_number[first_number.layer == 1][
        ["token_index", "region", "included"]]], ignore_index=True)
    diagnostic_region = diagnostics.merge(diagnostic_region, on="token_index", validate="one_to_many")
    diagnostic_region = diagnostic_region[diagnostic_region.included]
    depth_summary = diagnostic_region.groupby("region")[[
        "stabilization_depth", "late_stabilization_depth", "top1_switches"]].mean().reset_index()
    examples = examples.merge(depth_summary, on="region", how="left")
    reasoning_ids = sorted(frame.loc[frame.region == "reasoning", "token_index"].unique())
    mapping = {token: min(bins - 1, index * bins // len(reasoning_ids))
               for index, token in enumerate(reasoning_ids)}
    reasoning = frame[frame.region == "reasoning"].copy()
    reasoning["bin"] = reasoning.token_index.map(mapping)
    trajectory = reasoning.groupby(["layer", "bin"])[FEATURES].mean().reset_index()
    return examples, trajectory
