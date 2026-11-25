import json
from collections import defaultdict

import numpy as np
import pytest
import torch

from experiments.integrated_correction import answers, correction
from experiments.integrated_correction.answers import distribution, equivalent, extract, grade, parse_response
from experiments.integrated_correction.correction import (Assessment, CorrectionConfig, correct, displacement,
                                                          mixed_reward, rms, route)
from experiments.integrated_correction.data import CONSISTENCY_TAG, consistency_rows, group_id, split_problems, validate_partitions
from experiments.integrated_correction.learning import fit_probe, predict_probe, train_reward, weight_by_problem
from experiments.integrated_correction.model import (BudgetExhausted, Engine, Problem, prefix_positions,
                                                     reasoning_length, response_id, response_seed, sample_top_p)
from experiments.integrated_correction.stages import read_rollouts, release_bank, validate_responses
from experiments.integrated_correction.storage import Stage, digest, sha256, write


def label_bank(correct, wrong, explicit_refusals=0):
    values = [(str(i + 1), i == 0, "parsed", False)
              for i, n in enumerate([correct, *wrong]) for _ in range(n)]
    values += [(None, False, "refused", True)] * explicit_refusals
    values += [(None, False, "truncated", False)] * (30 - len(values))
    records = [{"id": str(i), "sample_index": i, "grading": {
        "answer": value, "correct": right, "status": status, "explicit_refusal": refusal}}
        for i, (value, right, status, refusal) in enumerate(values)]
    return distribution("gsm8k", records[::-1])


@pytest.mark.parametrize("n, label", [(14, False), (15, True)])
def test_label_boundaries(n, label):
    by_refusal = label_bank(0, [], n)
    by_correct = label_bank(n, [])
    by_wrong = label_bank(0, [n])
    assert by_refusal["labels"]["A"] is label
    assert by_correct["labels"]["B"] is label
    assert by_wrong["labels"]["C"] is label
    assert by_refusal["n_unique"] == 0
    assert sum(by_correct["failures"].values()) + n == 30


def test_label_overlap_subtypes_and_empty():
    assert not label_bank(11, [5])["labels"]["D"]
    assert label_bank(10, [5])["labels"]["D"]
    assert not label_bank(5, [4])["labels"]["D"]
    assert label_bank(5, [5])["labels"]["D"]
    assert not label_bank(0, [1] * 10)["labels"]["E"]
    assert label_bank(0, [1] * 11)["subtypes"]["E1"]
    assert label_bank(1, [1] * 10)["subtypes"]["E2"]
    assert label_bank(15, [15])["labels"] == {"A": False, "B": True, "C": True, "D": True, "E": False}
    assert label_bank(30, [])["subtypes"]["B1"]
    assert label_bank(0, [30])["subtypes"]["C1"]
    assert label_bank(4, [4])["unclassified"]
    assert label_bank(0, [0])["unclassified"]


def response(text, finish="eos"):
    return {"text": text, "finish": finish}


def test_strict_answer_and_end_status():
    assert extract("gsm8k", "12, not 15") is None
    assert extract("gsm8k", "#### 4\n#### +1,200.5000") == "1200.5"
    assert grade("gsm8k", response("calculated\n#### -0"), "0")["correct"]
    for value in ("#### 2 USD", "#### $5", "#### 1/2", "#### **5**", "#### 1,00", "#### 9\nexplanation"):
        assert extract("gsm8k", value) is None
    assert parse_response("gsm8k", response("I cannot answer."))["status"] == "refused"
    assert parse_response("gsm8k", response("1, 2, 3"))["status"] == "parse_failure"
    failed = grade("gsm8k", response("#### 4", "length"), "4")
    assert failed["status"] == "truncated" and not failed["correct"]
    assert not parse_response("gsm8k", response("Incomplete", "length"))["explicit_refusal"]


def test_math_nested_equivalence_and_failure(monkeypatch):
    assert extract("math", r"Earlier \boxed{0}. Final answer: \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert extract("math", r"\boxed{\frac{1}{2}") is None
    assert equivalent("math", r"\sqrt{8}", r"2\sqrt{2}") == "equal"
    assert grade("math", response(r"Final answer: \boxed{0.5}"), r"\frac{1}{2}")["correct"]
    assert equivalent("math", "not_math_???", "2") == "undecidable"
    assert label_bank(0, [2, 1])["n_unique"] == 2

    def raise_timeout(*_args, **_kwargs):
        raise answers.TimeoutException()

    monkeypatch.setattr(answers, "verify", raise_timeout)
    assert grade("math", response(r"\boxed{2}"), "2")["status"] == "timeout"


def test_sampling_and_rounded_positions():
    logits = torch.zeros(2, 200)
    result = sample_top_p(logits, torch.tensor([0.0, .9999]))
    expected = logits.sort(descending=True).indices
    assert result[0] == expected[0, 0]
    assert result[1] == expected[1, 189]
    assert prefix_positions(100) == [8, 20, 28, 40, 48, 60, 68, 80]
    assert prefix_positions(5) == [4]
    assert prefix_positions(2) == []
    p = "gsm8k/main/train/0"
    roles = ("label_bank", "initial", "feature_sample", "continuation", "corrected")
    seeds = [response_seed(42, p, role, i) for role in roles for i in range(30)]
    assert len(seeds) == len(set(seeds))
    assert seeds[0] == response_seed(42, p, roles[0], 0)
    assert seeds[0] != response_seed(42, p + "1", roles[0], 0)


def test_exact_split_and_id_validation():
    def row(i, question, official="train"):
        return {"id": str(i), "dataset": "math", "subset": "algebra", "level": "Level 1",
                "official_split": official, "group": group_id(question)}

    all_rows = [row(i, str(i)) for i in range(100)]
    all_rows.extend([row(100, " 5  "), row(101, " 8 "), row(102, "8", "test")])
    rows, removed = split_problems(all_rows, 42, None)
    assert removed == ["101", "8"]
    keyed = {r["id"]: r for r in rows}
    assert keyed["100"]["split"] == keyed["5"]["split"]
    counts = defaultdict(int)
    for r in rows:
        counts[r["split"]] += 1
    assert counts == {"fit": 80, "tune": 10, "calibration": 10, "test": 1}
    with pytest.raises(ValueError, match="leaks"):
        validate_partitions(rows + [{**keyed["100"], "id": "dup", "split": "test", "official_split": "test"}])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_partitions(rows + rows[:1])
    problem = keyed["5"]
    rid = response_id(problem["id"], "initial", 0)
    enriched = {"id": rid, "problem_id": problem["id"], "split": problem["split"], "role": "initial",
                "sample_index": 0, "seed": response_seed(42, problem["id"], "initial", 0)}
    assert validate_responses(problem, [enriched], 42, "initial", [0]) == [enriched]
    with pytest.raises(ValueError, match="provenance"):
        validate_responses(problem, [{**enriched, "split": "test"}], 42, "initial", [0])


def test_prefix_join_checks_fold_and_ids(tmp_path):
    p = {"id": "p", "split": "fit"}
    src = response_id("p", "initial", 0)
    ids = [response_id("p", "continuation", i) for i in range(4)]
    seeds = [response_seed(42, "p", "continuation", i) for i in range(4)]
    record = {"id": f"{src}/prefix/4", "source_response_id": src, "prefix_length": 4, "response_ids": ids,
              "seeds": seeds, "prefix_sha256": digest([0, 1, 2, 3, 4]), "attempts": 4, "successes": 2, "problem_id": "p", "split": "fit"}
    responses = [{"id": rid, "problem_id": "p", "split": "fit", "role": "continuation", "sample_index": i,
                  "seed": seeds[i], "grading": {"correct": i < 2}, "generation": {"prefix_length": 4},
                  "prompt_ids": [0], "output_ids": [1, 2, 3, 4, 5]}
                 for i, rid in enumerate(ids)]
    write(tmp_path / "responses.json", responses[::-1])
    write(tmp_path / "rollouts.json", [record])
    np.savez(tmp_path / "residuals.npz", residuals=np.ones((1, 1536)), ids=np.array([record["id"]]))
    x, y = read_rollouts(p, tmp_path, 42)
    assert x.shape == (1, 1536) and y.tolist() == [.5]
    write(tmp_path / "rollouts.json", [{**record, "split": "tune"}])
    with pytest.raises(ValueError, match="prefix provenance"):
        read_rollouts(p, tmp_path, 42)
    write(tmp_path / "rollouts.json", [record])
    write(tmp_path / "responses.json", responses + responses[:1])
    with pytest.raises(ValueError, match="Duplicate"):
        read_rollouts(p, tmp_path, 42)


def test_empty_reasoning_eos_and_answer_lines():
    get_text = lambda ids: "".join(chr(token) for token in ids)
    ids = list(map(ord, "#### 2")) + [0]
    assert reasoning_length("gsm8k", ids, {0}, get_text) == 0
    text = "derive\nFinal answer: \\boxed{8}"
    assert reasoning_length("math", list(map(ord, text)) + [0], {0}, get_text) == len("derive\n")
    assert reasoning_length("gsm8k", [65, 0, 66], {0}, get_text) == 1


def linear_head(value):
    return value[..., 0] - 2 * value[..., 1]


CAL = {"slope": 1., "intercept": 0.}
HEADS = {"global": (linear_head, CAL)}


def test_displacement_matches_finite_difference():
    vector = torch.tensor([.3, -.2])
    delta, gradient, value = displacement(vector, HEADS, {"global": 1.}, .1)
    assert rms(delta).item() == pytest.approx(.1 * rms(vector).item(), rel=1e-6)
    basis = torch.eye(2) * .0001
    a = (mixed_reward(vector + basis, HEADS, {"global": 1.}) + 1e-6).log()
    b = (mixed_reward(vector - basis, HEADS, {"global": 1.}) + 1e-6).log()
    finite = (a - b) / .0002
    assert torch.nn.functional.cosine_similarity(delta, finite, dim=0) > .999
    assert gradient == pytest.approx(float(rms(finite)), rel=.005)
    assert 0 < value < 1
    unstable = {"global": (lambda h: h[0] * torch.tensor(float("nan")), CAL)}
    assert displacement(vector, unstable, {"global": 1.}, .1) == (None, None, None)
    flat = {"global": (lambda h: h[0] * 0, CAL)}
    assert displacement(vector, flat, {"global": 1.}, .1)[0] is None


def test_routing_and_calibration():
    heads = {key: None for key in ("global", "B", "C", "E")}
    assert route({"B": .5, "C": .9, "D": 1., "E": .4}, heads) == {"B": .5 / 1.4, "C": .9 / 1.4}
    assert route({"D": 1., "B": None}, heads) == {"global": 1.}
    assert not route({}, {})
    assert fit_probe(([], []), ([], []), ([], []), minimum=30)["disabled"]
    x, y = ["first solution", "second response", "third solution", "fourth response"], [1, 0, 1, 0]
    probe = fit_probe((x, y), (x, y), (x, y), text=True)
    assert probe["c"] == .01
    restored = json.loads(json.dumps(probe))
    assert predict_probe(restored, "first solution") > predict_probe(restored, "fourth response")
    assert weight_by_problem(["a", "a", "b"]).tolist() == [.5, .5, 1.]


def test_reward_disable_and_train():
    vectors = np.zeros((4, 1536), dtype=np.float32)
    vectors[:, 0] = [-2., 2., -1., 1.]
    targets = np.array([0., 1., .25, .75], dtype=np.float32)
    ids = np.array(["a", "a", "b", "b"])
    pools = {fold: (vectors, targets, ids) for fold in ("fit", "tune", "calibration")}
    assert train_reward(pools, 30, "cpu", 42)[1]["disabled"]
    state, scores = train_reward(pools, 1, "cpu", 42)
    assert state and scores["calibration"]["status"] == "fitted"
    assert scores["calibration_metrics"]["bce"] < .50


class ToyEngine:
    eos_ids = {99}

    def __init__(self, eos_at=9):
        self.forward_tokens = self.generated_tokens = 0
        self.calls = []
        self.eos_at = eos_at

    def replay(self, ids, position, patch=None):
        self.forward_tokens += len(ids)
        self.calls.append(("replay", ids[:], patch[0] if patch else None))
        return torch.tensor([3. if patch else -2., -.5]), None

    def next_token(self, ids, patch=None):
        self.forward_tokens += len(ids)
        self.generated_tokens += 1
        self.calls.append(("next", ids[:], patch[0] if patch else None))
        return 99 if len(ids) >= self.eos_at else (3 if patch else 2)

    def record(self, problem, prompt, committed, role, index, seed):
        assert isinstance(problem, Problem)
        assert not hasattr(problem, "gold")
        return {"prompt_ids": prompt, "output_ids": committed, "role": role, "problem_id": problem.id}


def test_rollback_reinjects_and_discards_delta(monkeypatch):
    monkeypatch.setattr(correction, "MAX_INTERVENTIONS", 1)
    engine = ToyEngine()
    p = Problem("p", "gsm8k", "question")
    initial = {"prompt_ids": [1], "output_ids": [9]}
    assessment = Assessment(initial, .9, .3, {"B": .8})
    config = CorrectionConfig(engine, 42, .1, .7)
    fixed, logs = correct(p, assessment, HEADS, config)
    assert initial["output_ids"] == [9]
    assert fixed["output_ids"] == [3] * 4 + [2] * 4 + [99]
    assert len(logs) == 1 and logs[0]["accepted"]
    assert logs[0]["normal_tokens"] == [2] * 4 and logs[0]["alternative_tokens"] == [3] * 4
    patched = [ids for name, ids, pos in engine.calls if pos == 0 and name == "next"]
    assert patched == [[1] + [3] * n for n in range(4)]
    assert engine.calls[-1] == ("next", [1] + [3] * 4 + [2] * 4, None)
    ordinary, logs = correct(p, assessment, HEADS, CorrectionConfig(ToyEngine(), 42, 0., .7))
    assert ordinary["output_ids"] == [2] * 8 + [99] and not logs


def test_budget_and_atomic_resume(tmp_path):
    engine = Engine.__new__(Engine)
    engine.limit, engine.forward_tokens = 4, 0
    engine.count_forward(torch.zeros((2, 2)))
    with pytest.raises(BudgetExhausted):
        engine.count_forward(torch.zeros((1, 1)))
    assert engine.forward_tokens == 4

    stage = Stage(tmp_path, "example")
    assert stage.begin({"version": 1}, resume=False)

    def interrupted(out):
        write(out / "row.json", {"id": "broken"})
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        stage.unit("a", interrupted)
    assert not stage.shard("a").exists()
    resumed = Stage(tmp_path, "example")
    assert resumed.begin({"version": 1}, resume=True)
    directory = resumed.unit("a", lambda out: write(out / "row.json", {"id": "a"}))
    resumed.finish()
    assert not Stage(tmp_path, "example").begin({"version": 1}, resume=True)
    with pytest.raises(ValueError, match="Cannot resume"):
        Stage(tmp_path, "example").begin({"version": 2}, resume=True)
    write(directory / "row.json", {"id": "tampered"})
    with pytest.raises(ValueError, match="checksum"):
        Stage(tmp_path, "example").load()


def test_release_bank_checksums_id_join_and_missing_observations(tmp_path):
    directory = tmp_path / CONSISTENCY_TAG
    directory.mkdir()
    records = [{"id": f"train-{i}", "pid": i, "samples_pred": [2.] * 15 + [1.] * 15,
                "samples_format_ok": [True] * 29 + [False]} for i in (1, 2)]
    tokens = [{"id": f"train-{i}", "prompt_ids": [i], "output_ids": [3]} for i in (2, 1)]
    for name, data in (("types.jsonl", records), ("greedy_token_ids.jsonl", tokens)):
        (directory / name).write_text("".join(json.dumps(row) + "\n" for row in data))
    labels_path = directory / "types.jsonl"
    token_path = directory / "greedy_token_ids.jsonl"
    (directory / "SHA256SUMS.txt").write_text("".join(
        f"{sha256(path)}  {path.name}\n" for path in (labels_path, token_path)))
    config = {"release_dir": str(tmp_path)}
    joined, _ = consistency_rows(config, tmp_path)
    labels, greedy = joined[1]
    assert greedy["id"] == "train-1"
    row = {"id": "gsm8k/main/train/1", "gold": "2", "split": "fit",
           "release_bank": {"id": labels["id"], "samples": labels["samples_pred"],
                            "format_ok": labels["samples_format_ok"]}}
    samples = release_bank(row)
    validate_responses(row, samples, 42, "label_bank", list(range(30)), released=True)
    result = distribution("gsm8k", samples)
    assert result["labels"] == {"A": None, "B": True, "C": False, "D": True, "E": False}
    assert result["failures"] == {"release_format_failure": 1}
    assert samples[-1]["seed"] is None and samples[-1]["text"] is None
    assert samples[-1]["grading"]["answer"] is None
    token_path.write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        consistency_rows(config, tmp_path)
