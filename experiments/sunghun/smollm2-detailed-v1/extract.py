"""Gold-free detailed features from the model's own generated token path."""
import numpy as np
import torch

from common import confidence_stats
from gsm8k_lens.answers import token_regions


TOKEN_METRICS = ["generated_logprob", "entropy", "probability_margin", "top1_probability", "generated_rank"]
TRAJECTORY_METRICS = ["previous_norm", "hidden_norm", "delta_norm", "relative_delta", "cosine_distance"]
CHECKPOINT_METRICS = TOKEN_METRICS + ["js_to_final", "js_to_previous"]


def residual_metrics(before, after):
    before, after = before.float(), after.float()
    old, new = before.norm(dim=-1), after.norm(dim=-1)
    delta = (after - before).norm(dim=-1)
    cosine = 1 - ((before * after).sum(-1) / (old * new).clamp_min(1e-12)).clamp(-1, 1)
    return torch.stack([old, new, delta, delta / old.clamp_min(1e-12), cosine], -1)


def distribution(model, hidden):
    # Raw block states need exactly one frozen final RMSNorm before LM head.
    return model.lm_head(model.model.norm(hidden)).float().log_softmax(-1)


def summarize(logp, targets):
    top = logp.topk(2, dim=-1).values.exp()
    chosen = logp.gather(1, targets[:, None]).squeeze(1)
    rank = (logp > chosen[:, None]).sum(-1) + 1
    return torch.stack([chosen, -(logp.exp() * logp).sum(-1), top[:, 0] - top[:, 1],
                        top[:, 0], rank.float()], -1)


def js_divergence(a, b):
    mixture = torch.logaddexp(a, b) - np.log(2.)
    return (.5 * ((a.exp() * (a - mixture)).sum(-1) +
                  (b.exp() * (b - mixture)).sum(-1))).clamp_min(0)


@torch.inference_mode()
def extract(model, tokenizer, generation, config, quality=False):
    device = next(model.parameters()).device
    prompt, generated = generation["prompt_token_ids"], generation["generated_token_ids"]
    if not prompt or not generated:
        raise ValueError("Empty prompt/completion")
    text, regions = token_regions(tokenizer, generated)
    if text != generation["text"]:
        raise ValueError("Saved generation fails exact token/text roundtrip")
    whole = np.array([bool(r["text"].strip()) and r["token_id"] not in tokenizer.all_special_ids for r in regions])
    number = np.array([r["region"] == "final_number" and bool(r["text"].strip()) for r in regions])
    content = np.array([r["token_id"] not in tokenizer.all_special_ids for r in regions])
    n, depth = len(generated), len(model.model.layers)
    times = [t for t in config["extraction"]["checkpoints"] if t < n]
    p = len(prompt)
    before = torch.arange(p - 1, p - 1 + n, device=device)
    after = before + 1
    cp = before[times]
    prefix_count = min(config["extraction"]["readout_prefix_tokens"], n)
    readout_layers = config["extraction"]["readout_layers"]
    raw, pooled, trajectories, selected = {}, {}, {}, {}
    whole_t, number_t = torch.as_tensor(whole, device=device), torch.as_tensor(number, device=device)

    def collect(layer):
        def hook(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            hidden = value[0]
            raw[layer] = hidden[cp].float().clone()
            if layer == 0:
                return
            states = hidden[after]
            zero = torch.zeros(hidden.shape[-1], dtype=torch.float32, device=device)
            pooled[layer] = torch.stack([
                states[int(np.flatnonzero(content)[-1])].float() if content.any() else zero,
                states[whole_t].float().mean(0) if whole.any() else zero,
                states[number_t].float().mean(0) if number.any() else zero,
            ]).cpu().numpy()
            trajectories[layer] = residual_metrics(args[0][0, before], hidden[before]).cpu().numpy()
            if layer in readout_layers or layer == depth:
                count = n if layer == depth else prefix_count
                selected[layer] = hidden[before[:count]].float().clone()
        return hook

    handles = [model.model.embed_tokens.register_forward_hook(collect(0))]
    handles += [block.register_forward_hook(collect(i)) for i, block in enumerate(model.model.layers, 1)]
    try:
        model.model(input_ids=torch.tensor([prompt + generated], device=device), use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    if set(raw) != set(range(depth + 1)):
        raise ValueError("Missing block capture")

    # Canonical prompt-only states prevent completion-shaped numerical differences at t=0.
    canonical = {}
    def canonical_hook(layer):
        def hook(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            canonical[layer] = value[0, -1].float().clone()
        return hook
    handles = [model.model.embed_tokens.register_forward_hook(canonical_hook(0))]
    handles += [block.register_forward_hook(canonical_hook(i)) for i, block in enumerate(model.model.layers, 1)]
    try:
        model.model(input_ids=torch.tensor([prompt], device=device), use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    relative_error = max(float((raw[i][0] - canonical[i]).norm() / canonical[i].norm().clamp_min(1e-12)) for i in raw)
    if relative_error > .001:
        raise ValueError("Prompt causality discrepancy exceeds tolerance")
    for layer in raw:
        raw[layer][0] = canonical[layer]
        if layer in selected:
            selected[layer][0] = canonical[layer]
    for layer in trajectories:
        trajectories[layer][0] = residual_metrics(canonical[layer - 1], canonical[layer]).cpu().numpy()

    chunk = config["extraction"]["projection_chunk_tokens"]
    targets = torch.tensor(generated, device=device)
    topk = config["extraction"]["top_k"]
    native, native_ids, native_logp = [], [], []
    for start in range(0, n, chunk):
        lp = distribution(model, selected[depth][start:start + chunk])
        native.append(summarize(lp, targets[start:start + chunk]).cpu().numpy())
        top = lp.topk(topk, -1)
        native_ids.append(top.indices.cpu().numpy().astype(np.int32))
        native_logp.append(top.values.cpu().numpy())
    native = np.concatenate(native)
    prefix = []
    for layer in readout_layers:
        pieces = []
        for start in range(0, prefix_count, chunk):
            lp = distribution(model, selected[layer][start:min(prefix_count, start + chunk)])
            pieces.append(summarize(lp, targets[start:start + len(lp)]).cpu().numpy())
        prefix.append(np.concatenate(pieces))

    checkpoint_stats, checkpoint_ids, checkpoint_logp = [], [], []
    final = distribution(model, raw[depth])
    previous = distribution(model, raw[0])
    for layer in range(1, depth + 1):
        lp = distribution(model, raw[layer])
        basic = summarize(lp, targets[times])
        values = torch.cat([basic, js_divergence(lp, final)[:, None], js_divergence(lp, previous)[:, None]], -1)
        checkpoint_stats.append(values.cpu().numpy())
        top = lp.topk(topk, -1)
        checkpoint_ids.append(top.indices.cpu().numpy().astype(np.int32))
        checkpoint_logp.append(top.values.cpu().numpy())
        previous = lp

    confidence, names = [], []
    for name, mask in [("whole", whole), ("number", number)]:
        columns, stats = confidence_stats(native[mask, 0], native[mask, 1], native[mask, 2])
        names += [name + "_" + c for c in columns]
        confidence.extend(stats)
    qa = dict(prompt_full_vs_prefix_relative_l2=relative_error, positions=len(times),
              matches=int(sum(np.concatenate(native_ids)[t, 0] == generated[t] for t in times)),
              independent_prefix_checks=0, independent_prefix_max_abs=None)
    if quality:
        errors = []
        for t in times:
            captured = []
            def short_hook(module, args, output):
                value = output[0] if isinstance(output, tuple) else output
                captured.append(value[0, -1].float().clone())
            mid = config["extraction"]["primary_layer"]
            handle = model.model.layers[mid - 1].register_forward_hook(short_hook)
            try:
                model.model(input_ids=torch.tensor([prompt + generated[:t]], device=device), use_cache=False, return_dict=True)
            finally:
                handle.remove()
            actual = raw[mid][times.index(t)]
            errors.append(float((actual - captured[0]).abs().max()))
            if not torch.allclose(actual, captured[0], atol=5e-4, rtol=5e-4):
                raise ValueError("Independent truncated-prefix check failed")
        qa.update(independent_prefix_checks=len(errors), independent_prefix_max_abs=max(errors))
    checks = generation.get("generation_quality", {}).get("generated_logprob")
    if checks is not None:
        error = float(np.abs(native[:, 0] - checks).max())
        qa["generation_replay_logprob_max_error"] = error
        if error > .05:
            raise ValueError("Original generation/replay logprob QA failed")
    features = dict(checkpoints=np.asarray(times, np.int16),
        hidden=np.stack([raw[i].cpu().numpy() for i in range(depth + 1)], 1),
        pooled_hidden=np.stack([pooled[i] for i in range(1, depth + 1)]),
        trajectory=np.stack([trajectories[i] for i in range(1, depth + 1)], 1),
        checkpoint_readouts=np.stack(checkpoint_stats, 1),
        checkpoint_top_ids=np.stack(checkpoint_ids, 1), checkpoint_top_logprob=np.stack(checkpoint_logp, 1),
        prefix_readouts=np.stack(prefix, 1), token_readouts=native,
        token_top_ids=np.concatenate(native_ids), token_top_logprob=np.concatenate(native_logp),
        token_region=np.asarray([{"excluded": 0, "reasoning": 1, "final_number": 2, "format": 3, "unparsed": 4}[r["region"]] for r in regions], np.int8),
        token_offsets=np.asarray([[r["char_start"], r["char_end"]] for r in regions], np.int32),
        token_masks=np.stack([content, whole, number], -1), confidence=np.asarray(confidence, np.float32),
        length=np.log1p([p, n, int(whole.sum()), int(number.sum())]).astype(np.float32))
    for name, value in features.items():
        if name != "confidence" and not np.isfinite(value).all():
            raise ValueError("Nonfinite " + name)
    return features, qa, names
