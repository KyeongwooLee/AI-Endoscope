"""Gold-free assessment, routing, and no-cache hidden correction."""

from dataclasses import dataclass

import numpy as np
import torch

from .answers import CLASSES, cluster, equivalent, parse_response
from .learning import predict_probe, reward_values
from .model import MAX_NEW, BudgetExhausted

BLOCK = 4
MAX_INTERVENTIONS = 32
ALPHAS = (.02, .1, .4)
REWARD_THRESHOLDS = (.3, .5, .7)


@dataclass(frozen=True)
class Assessment:
    initial: dict
    risk: float | None
    threshold: float | None
    probabilities: dict

    @property
    def triggered(self):
        return self.risk is not None and self.threshold is not None and self.risk >= self.threshold


@dataclass(frozen=True)
class CorrectionConfig:
    engine: object
    seed: int
    alpha: float
    threshold: float
    sample_index: int = 0


def assess(initial, vector, bundle):
    probes = bundle["probes"]
    return Assessment(initial, predict_probe(probes["r_mean/incorrect"], vector),
                      bundle["risk_threshold"], class_probabilities(probes, "r_mean", vector))


def class_probabilities(probes, view, vector):
    return {name: predict_probe(probes[f"{view}/{name}"], vector) for name in CLASSES}


def route(probabilities, heads):
    selected = {k: probabilities[k] for k in CLASSES if k in heads and probabilities.get(k) is not None
                and probabilities[k] >= .5}
    if not selected:
        return {"global": 1.} if "global" in heads else {}
    total = sum(selected.values())
    return {key: value / total for key, value in selected.items()}


def mixed_reward(residual, heads, weights):
    return sum(weights[key] * reward_values(head, residual, calibration)
               for key, (head, calibration) in heads.items() if key in weights)


def rms(tensor):
    return tensor.square().mean().sqrt()


def displacement(residual, heads, weights, alpha):
    with torch.enable_grad():
        h = residual.detach().float().requires_grad_(True)
        value = mixed_reward(h, heads, weights)
        if not value.requires_grad or not torch.isfinite(value):
            return None, None, None
        g, = torch.autograd.grad(torch.log(value + 1e-6), h, allow_unused=True)
    if g is None or not torch.isfinite(g).all():
        return None, None, float(value.detach())
    magnitude = rms(g)
    if not torch.isfinite(magnitude) or magnitude <= 0:
        return None, None, float(value.detach())
    delta = alpha * rms(h.detach()) * g / (magnitude + 1e-8)
    if not torch.isfinite(delta).all():
        return None, float(magnitude), float(value.detach())
    return delta, float(magnitude), float(value.detach())


def block(engine, prefix, count, patch=None):
    tokens = []
    for _ in range(count):
        token = engine.next_token(prefix + tokens, patch=patch)
        tokens.append(token)
        if token in engine.eos_ids:
            break
    return tokens


@torch.no_grad()
def end_reward(engine, ids, heads, weights, patch=None):
    hidden, _ = engine.replay(ids, -1, patch=patch)
    value = float(mixed_reward(hidden, heads, weights))
    return value if np.isfinite(value) else None


def regenerate(problem, assessment, heads, config, weights):
    if not assessment.triggered:
        return assessment.initial, []
    engine = config.engine
    prompt = assessment.initial["prompt_ids"]
    committed, logs = [], []
    attempts = 0
    try:
        while len(committed) < MAX_NEW[problem.dataset]:
            count = min(BLOCK, MAX_NEW[problem.dataset] - len(committed))
            prefix = prompt + committed
            start_cost = engine.forward_tokens, engine.generated_tokens
            normal = block(engine, prefix, count)
            selected = normal
            if weights and config.alpha != 0 and attempts < MAX_INTERVENTIONS:
                reward = end_reward(engine, prefix + normal, heads, weights)
                event = {"position": len(committed), "class_probabilities": assessment.probabilities,
                         "weights": weights, "reward_before": reward, "reward_after": None,
                         "start_reward": None, "gradient_rms": None, "delta_rms": None, "accepted": False,
                         "attempted": False, "normal_tokens": normal, "alternative_tokens": None}
                if reward is not None and reward < config.threshold:
                    attempts += 1
                    event["attempted"] = True
                    h, _ = engine.replay(prefix, -1)
                    delta, gradient_rms, start_reward = displacement(h, heads, weights, config.alpha)
                    event.update(gradient_rms=gradient_rms, start_reward=start_reward)
                    if delta is not None:
                        event["delta_rms"] = float(rms(delta))
                        patch = (len(prefix) - 1, delta)
                        alternative = block(engine, prefix, count, patch)
                        alternative_reward = end_reward(engine, prefix + alternative, heads, weights, patch)
                        event.update(reward_after=alternative_reward, alternative_tokens=alternative)
                        if alternative_reward is not None and alternative_reward > reward:
                            selected = alternative
                            event["accepted"] = True
                event["forward_tokens"] = engine.forward_tokens - start_cost[0]
                event["generated_tokens"] = engine.generated_tokens - start_cost[1]
                logs.append(event)
            committed.extend(selected)
            if committed[-1] in engine.eos_ids:
                break
    except BudgetExhausted as error:
        error.logs = logs
        raise
    response = engine.record(problem, prompt, committed, "corrected", config.sample_index, config.seed)
    return response, logs


def correct(problem, probe_bundle, reward_bundle, config):
    """The input assessment contains only the generated response and fixed predictions."""
    return regenerate(problem, probe_bundle, reward_bundle, config,
                      route(probe_bundle.probabilities, reward_bundle))


def vote(problem, responses, weights):
    representatives, totals, first = [], {}, {}
    for response, weight in zip(responses, weights):
        parsed = parse_response(problem.dataset, response)
        answer = parsed["answer"]
        if parsed["status"] != "parsed" or equivalent(problem.dataset, answer, answer) != "equal":
            continue
        idx, failure = cluster(problem.dataset, representatives, answer)
        if failure is None:
            first.setdefault(idx, response)
            totals[idx] = totals.get(idx, 0.) + weight
    # The first generated answer wins ties, with the greedy response first.
    return first[max(totals, key=totals.get)] if totals else responses[0]
