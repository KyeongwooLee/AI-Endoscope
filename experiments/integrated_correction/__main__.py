"""Run one stage with the same immutable experiment config and run directory."""

import argparse
import fcntl
import os
import re
from pathlib import Path

from .data import RELEASES, consistency_assets
from .evaluation import eval_stage, tune_stage
from .reporting import report_stage
from .stages import (distributions_stage, features_stage, label_bank_stage, prepare_stage,
                     probes_stage, rewards_stage, rollouts_stage)
from .storage import Stage, environment, read, sha256, source

PREPARE = "prepare"
BANK = "generate-label-bank"
LABEL = "label-distributions"
FEATURES = "generate-features"
PROBE = "train-probes"
ROLLOUT = "generate-prefix-rollouts"
REWARD = "train-rewards"
TUNE = "tune-correction"
EVAL = "evaluate"

STAGES = {
    PREPARE: (prepare_stage, ()),
    BANK: (label_bank_stage, (PREPARE,)),
    LABEL: (distributions_stage, (PREPARE, BANK)),
    FEATURES: (features_stage, (PREPARE,)),
    PROBE: (probes_stage, (PREPARE, LABEL, FEATURES)),
    ROLLOUT: (rollouts_stage, (PREPARE, FEATURES)),
    REWARD: (rewards_stage, (PREPARE, LABEL, ROLLOUT)),
    TUNE: (tune_stage, (PREPARE, FEATURES, PROBE, REWARD)),
    EVAL: (eval_stage, (PREPARE, PROBE, REWARD, TUNE)),
    "report": (report_stage, (PREPARE, BANK, LABEL, FEATURES, PROBE, ROLLOUT, REWARD, TUNE, EVAL)),
}


def config_at(path):
    config = read(path)
    config.setdefault("gsm8k_release_bank", False)
    template = read(Path(__file__).with_name("config.json"))
    if config.keys() != template.keys():
        raise ValueError(f"Config keys must match the shared config: {sorted(template)}")
    if type(config["gsm8k_release_bank"]) is not bool:
        raise ValueError("gsm8k_release_bank must be a boolean")
    if config["seed"] != 42:
        raise ValueError("The planned global seed is 42")
    for key in ("model_revision", "gsm8k_revision", "math_revision"):
        if key == "math_revision" and config[key] is None:
            continue
        if not isinstance(config[key], str) or not re.fullmatch(r"[0-9a-f]{40}", config[key]):
            raise ValueError(f"{key} must be a full pinned SHA")
    if not isinstance(config["model_weights_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", config["model_weights_sha256"]):
        raise ValueError("Expected the model weights SHA-256")
    limit = config["limit_per_dataset"]
    if limit is not None and (type(limit) is not int or limit < 2):
        raise ValueError("limit_per_dataset must be null or an integer of at least 2")
    if type(config["forward_budget_per_problem"]) is not int or config["forward_budget_per_problem"] <= 0:
        raise ValueError("forward_budget_per_problem must be a positive integer")
    if config["release_dir"] is not None:
        config["release_dir"] = str(Path(config["release_dir"]).expanduser().resolve())
    if config["dtype"] not in ("bfloat16", "float32"):
        raise ValueError("dtype must be bfloat16 or float32")
    return config


def execute(args):
    config = config_at(args.config)
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    cache = Path(os.environ.get("INTEGRATED_CORRECTION_CACHE", root / "cache"))
    snapshot = {"config": config, "source": source(), "environment": environment(config["device"])}
    with (root / ".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute_locked(args, root, config, cache, snapshot)


def execute_locked(args, root, config, cache, snapshot):
    stage = Stage(root, args.stage)
    run, required = STAGES[args.stage]
    deps = {name: Stage(root, name) for name in required}
    inputs = {}
    for name, dep in deps.items():
        receipt = dep.load()
        if any(receipt["inputs"][k] != v for k, v in snapshot.items()):
            raise ValueError(f"Incompatible config, code or environment in {name}")
        inputs[name] = sha256(dep.receipt)
    if args.stage == PREPARE and config["release_dir"]:
        sources = [RELEASES["gsm8k"]] if config["gsm8k_release_bank"] else RELEASES.values()
        for tag, asset, _ in sources:
            for name in (asset, "SHA256SUMS.txt"):
                path = Path(config["release_dir"]) / tag / name
                inputs[f"release/{tag}/{name}"] = sha256(path)
    if args.stage == PREPARE and config["gsm8k_release_bank"]:
        for path in consistency_assets(config, cache):
            inputs[f"consistency/{path.name}"] = sha256(path)
    identity = {**snapshot, "stage": args.stage, "dependencies": inputs}
    if stage.begin(identity, args.resume):
        print(f"{args.stage} started", flush=True)
        run(stage, config, cache, deps)
        stage.finish()
    print(f"{args.stage} complete — {len(stage.units)} units — {stage.receipt}", flush=True)
    if args.stage == "report":
        print(stage.read_unit("all") / "report.md", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        execute(args)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
