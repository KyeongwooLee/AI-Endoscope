"""Exercise artifact joins and locked evaluation with deterministic toy generation."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.integrated_correction import __main__ as cli, stages
from experiments.integrated_correction.__main__ import STAGES, execute
from experiments.integrated_correction.model import HIDDEN_SIZE, BudgetExhausted, Engine, response_id, response_seed
from experiments.integrated_correction.storage import Stage, read, write


class FixtureEngine:
    eos_ids = {9}
    model = SimpleNamespace(config=SimpleNamespace(hidden_size=HIDDEN_SIZE))

    def prompt(self, problem):
        return [7, 3 if problem.question == "matched" else 4]

    def start_cost(self, limit=None):
        self.limit = limit
        self.forward_tokens = self.generated_tokens = 0

    def cost(self):
        return {"forward_tokens": self.forward_tokens, "generated_tokens": self.generated_tokens,
                "seconds": 0., "peak_gpu_bytes": 0}

    def pay(self, forward, generated=0):
        if self.limit is not None and self.forward_tokens + forward > self.limit:
            raise BudgetExhausted
        self.forward_tokens += forward
        self.generated_tokens += generated

    def record(self, p, prompt, tokens, role, index, seed, prefix=()):
        answer = tokens[-2]
        text = f"Work through it.\n#### {answer}" if p.dataset == "gsm8k" else f"Work. Final answer: \\boxed{{{answer}}}"
        return {"id": response_id(p.id, role, index), "problem_id": p.id, "role": role,
                "sample_index": index, "seed": response_seed(seed, p.id, role, index),
                "prompt_ids": prompt, "output_ids": tokens, "text": text, "finish": "eos",
                "generation": {"prefix_length": len(prefix)}}

    def generate_many(self, p, prompt, role, indices, seed, sampled, prefix=(), **sampling):
        if sampling:
            assert role == "label_bank" and sampling == {"temperature": 1., "top_p": 1.}
        return self.generate_requests(p, prompt, role, [(i, prefix) for i in indices], seed, sampled)

    def generate_requests(self, p, prompt, role, requests, seed, sampled=True):
        self.pay(len(requests) * 10, len(requests) * 6)
        answers = [2 if (i % 2 == 0 if sampled else p.question == "matched") else 1 for i, _ in requests]
        return [self.record(p, prompt, [*prefix, *([5] * 4), answer, 9], role, i, seed, prefix)
                for (i, prefix), answer in zip(requests, answers)]

    def next_token(self, ids, patch=None):
        self.pay(len(ids), 1)
        return 9 if len(ids) == 7 else 2 if len(ids) == 6 else 5

    def replay(self, ids, positions, patch=None):
        self.pay(len(ids))
        vector = torch.ones(HIDDEN_SIZE) * (1. if ids[1] == 3 else -1.)
        return (vector.repeat(len(positions), 1) if isinstance(positions, list) else vector), None

    def features(self, p, response, confidence=False):
        self.pay(len(response["prompt_ids"]) + len(response["output_ids"]))
        vector = np.full(HIDDEN_SIZE, response["output_ids"][-2] - 1.5, dtype=np.float32)
        features = {"r_mean": vector, "q_end": vector.copy()}
        if confidence:
            features["confidence"] = np.array([vector[0], len(response["output_ids"])])
        return features, 4


@pytest.mark.parametrize("use_release", [False, True])
def test_cli_locks_test_and_reports_both_cost_regimes(tmp_path, monkeypatch, use_release):
    rows = []
    for fold in ("fit", "tune", "calibration", "test"):
        for dataset, question in (("gsm8k", "matched"), ("gsm8k" if use_release else "math", "missed")):
            key = f"{dataset}/{question}/{fold}"
            rows.append({"id": key, "dataset": dataset, "question": question, "gold": "2",
                         "group": key, "split": fold, "official_split": "test" if fold == "test" else "train"})
            if use_release and fold != "test":
                rows[-1]["release_bank"] = {"id": key, "samples": [2.] * 15 + [1.] * 15,
                                           "format_ok": [True] * 30,
                                           "prompt_ids": FixtureEngine().prompt(SimpleNamespace(question=question)),
                                           "output_ids": [5] * 4 + [2 if question == "matched" else 1, 9]}
    monkeypatch.setattr(cli, "consistency_assets", lambda *_: ())
    monkeypatch.setattr(stages, "prepare", lambda *_: (rows, {
        "missing_gold": [], "removed_train_test_duplicates": [],
        "gsm8k_release_bank": {"temperature": 1.} if use_release else None}))
    monkeypatch.setattr(Engine, "load", lambda *_: FixtureEngine())
    config = read(Path(__file__).parents[1] / "config.json")
    config.update(device="cpu", dtype="float32", limit_per_dataset=4, forward_budget_per_problem=30,
                  gsm8k_release_bank=use_release)
    config_path = tmp_path / "config.json"
    write(config_path, config)
    root = tmp_path / "run"

    def run(stage):
        execute(SimpleNamespace(config=config_path, run_dir=root, stage=stage, resume=True))

    for stage in STAGES:
        run(stage)
    bank = Stage(root, "generate-label-bank")
    bank.load()
    assert not any("/test" in key for key in bank.units)
    tune = Stage(root, "tune-correction")
    tune.load()
    locked = read(tune.read_unit("locked") / "locked.json")
    assert len(locked["settings_grid"]) == 9 and locked["tune_problems"] == 2
    assert locked["risk_threshold"] is not None
    report = Stage(root, "report")
    report.load()
    values = read(report.read_unit("all") / "report.json")["results"]
    assert values["combined"]["test_problems"] == 2
    study = values["gsm8k" if use_release else "math"]
    baseline = .5 if use_release else 0.
    assert study["fixed"]["greedy"]["accuracy"] == baseline
    assert study["fixed"]["routed"]["accuracy"] == 1.
    assert study["budget"]["routed"]["accuracy"] == baseline
    assert study["budget"]["routed"]["budget_exhausted"] == 1
    assert study["budget"]["routed"]["forward_tokens"] <= 30 * study["test_problems"]
    if use_release:
        assert "math" not in values
        assert study["label_distributions"]["fit"]["unknown_types"]["A"] == 2
        features = Stage(root, "generate-features")
        features.load()
        key = next(iter(features.units))
        generated = read(features.read_unit(key) / "responses.json")
        assert len(generated) == 1 and generated[0]["seed"] is None
        rollouts = Stage(root, "generate-prefix-rollouts")
        rollouts.load()
        assert (rollouts.read_unit(key) / "sampled_source.json").exists()
    assert values["combined"]["reward_test"]["global"]["n"] > 0
    run("report")
