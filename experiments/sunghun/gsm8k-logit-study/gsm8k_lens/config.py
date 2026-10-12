from pathlib import Path
from .storage import read_json, digest


def load_config(path):
    cfg = read_json(path)
    required = {"schema_version", "seed", "model", "dataset", "prompt", "generation",
                "extraction", "quality", "analysis"}
    if set(cfg) != required or cfg["schema_version"] != 1:
        raise ValueError("Unsupported config keys/schema")
    if cfg["model"]["dtype"] not in ("bfloat16", "float32"):
        raise ValueError("Only bfloat16 and float32 are supported")
    if cfg["model"]["attention_backend"] not in ("sdpa", "eager"):
        raise ValueError("Use sdpa or eager attention")
    gen, ext, ds, ana, quality = (cfg[k] for k in
                                 ("generation", "extraction", "dataset", "analysis", "quality"))
    if gen["batch_size"] != 1:
        raise ValueError("This experiment requires batch_size=1")
    for key in ("max_new_tokens", "max_prompt_tokens", "max_sequence_tokens"):
        if gen[key] < 1:
            raise ValueError(f"Invalid {key}")
    if gen["max_prompt_tokens"] + gen["max_new_tokens"] > gen["max_sequence_tokens"]:
        raise ValueError("max_sequence_tokens must cover prompt + generation")
    if ext["cache_mode"] not in ("full", "features_only") or ext["top_k"] != 5:
        raise ValueError("Use cache_mode full/features_only and top_k=5")
    if ext["projection_chunk_tokens"] < 1 or ext["minimum_free_disk_gib"] < 0:
        raise ValueError("Invalid extraction memory/storage setting")
    if ds["pilot_size"] < 1 or ds["exploration_size"] < 1:
        raise ValueError("Pilot and exploration must be nonempty")
    if ds["pilot_size"] + ds["exploration_size"] > ds["expected_train_size"]:
        raise ValueError("Pilot/exploration overlap or exceed train size")
    if not 1 <= quality["generation_replay_examples"] <= ds["pilot_size"]:
        raise ValueError("Quality samples must fit the pilot")
    if not 0 <= quality["generation_replay_min_top1_agreement"] <= 1:
        raise ValueError("Invalid agreement threshold")
    if min(quality["native_lens_max_logprob_error"],
           quality["generation_replay_max_logprob_error"]) < 0:
        raise ValueError("Invalid numerical tolerance")
    for key in ("primary_layer", "late_start_layer"):
        if not 1 <= ana[key] <= cfg["model"]["expected_layers"]:
            raise ValueError(f"Invalid {key}")
    if ana["bootstrap_replicates"] < 2 or ana["trajectory_bins"] < 1:
        raise ValueError("Invalid analysis sampling parameters")
    if cfg["prompt"]["user_template"].count("{question}") != 1:
        raise ValueError("Prompt must contain exactly one {question}")
    return cfg


def manifest_for_run(run, cfg):
    manifest = read_json(Path(run) / "manifest.json")
    if manifest["config_hash"] != digest(cfg):
        raise ValueError("Config changed. Preserve this run and use a new --run directory.")
    return manifest
