"""Opt-in tests of exact residual and logit alignment on the pinned weights."""

import os
from pathlib import Path

import pytest
import torch

from experiments.integrated_correction import correction
from experiments.integrated_correction.correction import Assessment, CorrectionConfig, block, correct
from experiments.integrated_correction.model import MAX_NEW, Engine, Problem
from experiments.integrated_correction.storage import read


@pytest.fixture(scope="module")
def qwen():
    if os.environ.get("INTEGRATED_CORRECTION_TEST_QWEN") != "1":
        pytest.skip("Set INTEGRATED_CORRECTION_TEST_QWEN=1 to run the pinned CUDA model tests")
    config = read(Path(__file__).parents[1] / "config.json")
    return Engine.load(config)


@torch.no_grad()
def test_pinned_qwen_residual_and_next_token(qwen):
    p = Problem("alignment", "gsm8k", "What is 2 plus 3?")
    qwen.start_cost()
    prompt = qwen.prompt(p)
    response, = qwen.generate_many(p, prompt, "initial", [0], 42, False)
    tokens = response["output_ids"]
    assert response["finish"] == "eos" and len(tokens) > 2
    ids = prompt + tokens[:3]
    output = qwen.model(torch.tensor([ids], device=qwen.device), use_cache=False, output_hidden_states=True)
    positions = [len(prompt) - 1, len(prompt), len(prompt) + 1]
    captured, readout = qwen.replay(ids, positions, readout=positions)
    torch.testing.assert_close(captured, output.hidden_states[20][0, positions].float(), rtol=0, atol=0)
    expected = output.logits[0, positions].argmax(-1).tolist()
    assert qwen.model.lm_head(readout).argmax(-1).tolist() == expected == tokens[:3]
    assert qwen.next_token(prompt) == tokens[0]

    # The hook must be removed after every call, including a patched replay.
    delta = torch.ones_like(captured[0])
    changed, _ = qwen.replay(ids, positions, patch=(positions[0], delta))
    assert not torch.equal(changed[0], captured[0])
    torch.testing.assert_close(qwen.replay(ids, positions)[0], captured, rtol=0, atol=0)


def test_pinned_qwen_alpha_zero_and_empty_reasoning(qwen):
    p = Problem("alpha0", "math", "Calculate 2+3.")
    qwen.start_cost()
    prompt = qwen.prompt(p)
    raw, = qwen.generate_many(p, prompt, "initial", [0], 42, False)
    assets = {"global": (lambda h: h[0], {"slope": 1., "intercept": 0.})}
    response, logs = correct(p, Assessment(raw, 1., .5, {}),
                             assets, CorrectionConfig(qwen, 42, 0., .5))
    reference = block(qwen, prompt, MAX_NEW[p.dataset])
    assert not logs
    assert response["output_ids"] == reference
    stub = qwen.record(p, prompt, [min(qwen.eos_ids)], "initial", 0, 42)
    features, count = qwen.features(p, stub)
    assert count == 0
    assert (features["q_end"] == features["r_mean"]).all()


def test_pinned_qwen_nonzero_intervention(qwen, monkeypatch):
    p = Problem("intervention", "math", "Calculate 2+3.")
    qwen.start_cost()
    prompt = qwen.prompt(p)
    initial = qwen.record(p, prompt, [], "initial", 0, 42)
    heads = {"global": (lambda h: h.mean() * .01, {"slope": 1., "intercept": 0.})}
    monkeypatch.setattr(correction, "MAX_INTERVENTIONS", 1)
    monkeypatch.setattr(correction, "MAX_NEW", {"math": 8})
    response, logs = correct(p, Assessment(initial, 1., .5, {}), heads,
                             CorrectionConfig(qwen, 42, .1, .99))
    assert len(logs) == 1 and logs[0]["attempted"]
    assert logs[0]["gradient_rms"] > 0 and logs[0]["delta_rms"] > 0
    assert logs[0]["alternative_tokens"]
    assert logs[0]["accepted"] == (logs[0]["reward_after"] > logs[0]["reward_before"])
    assert response["role"] == "corrected" and not initial["output_ids"]
