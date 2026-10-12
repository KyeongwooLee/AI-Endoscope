import json
from pathlib import Path
import random
import shutil
import time
import traceback

from .answers import label_answer, parse_answer, token_regions
from .config import manifest_for_run
from .storage import (atomic_write, commit_files, complete, digest, file_digest, read_json,
                      read_jsonl, source_digest, write_json, write_jsonl, write_parquet)


PHASES = ("pilot", "exploration", "test")


def prepare(cfg, run):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    from .engine import environment

    run = Path(run)
    if (run / "manifest.json").exists():
        manifest = manifest_for_run(run, cfg)
        for relative, checksum in manifest["data_files"].items():
            if file_digest(run / relative) != checksum:
                raise ValueError(f"Prepared data changed: {relative}")
        print("Prepared run already exists; preserving revisions and sample IDs.", flush=True)
        return
    api = HfApi()
    model_revision = api.model_info(cfg["model"]["id"], revision=cfg["model"]["revision"]).sha
    data_revision = api.dataset_info(cfg["dataset"]["id"], revision=cfg["dataset"]["revision"]).sha
    ds = load_dataset(cfg["dataset"]["id"], cfg["dataset"]["name"], revision=data_revision)
    for split in ("train", "test"):
        expected = cfg["dataset"][f"expected_{split}_size"]
        if len(ds[split]) != expected:
            raise ValueError(f"Unexpected {split} size: {len(ds[split])}, expected {expected}")
    order = list(range(len(ds["train"])))
    random.Random(cfg["seed"]).shuffle(order)
    pilot, explore = cfg["dataset"]["pilot_size"], cfg["dataset"]["exploration_size"]
    selected = {"pilot": ("train", order[:pilot]),
                "exploration": ("train", order[pilot:pilot + explore]),
                "test": ("test", list(range(len(ds["test"]))))}
    gold, sample_ids, files = {}, {}, []
    for phase, (split, indexes) in selected.items():
        rows = []
        for index in indexes:
            row = ds[split][index]
            if parse_answer(row["answer"])["status"] != "ok":
                raise ValueError(f"Invalid gold answer: {split}:{index}")
            sample_id = f"{split}-{index:05d}"
            rows.append({"id": sample_id, "question": row["question"], "split": split, "index": index})
            gold[sample_id] = row["answer"]
        relative = f"data/{phase}.jsonl"
        write_jsonl(run / relative, rows)
        files.append(relative)
        sample_ids[phase] = [row["id"] for row in rows]
    write_json(run / "data/gold.json", gold)
    write_json(run / "sample_ids.json", sample_ids)
    files += ["data/gold.json", "sample_ids.json"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["id"], revision=model_revision,
                                               use_fast=True, trust_remote_code=False)
    write_json(run / "manifest.json", {
        "config": cfg, "config_hash": digest(cfg), "model_revision": model_revision,
        "dataset_revision": data_revision, "chat_template": tokenizer.chat_template,
        "chat_template_hash": digest(tokenizer.chat_template), "environment": environment(),
        "source_hash": source_digest(), "created_unix": time.time(),
        "data_files": {relative: file_digest(run / relative) for relative in files},
    })
    print(f"Prepared {sum(map(len, sample_ids.values()))} problems at {run}", flush=True)


def rows_for_phase(run, manifest, phase, limit=None):
    relative = f"data/{phase}.jsonl"
    if file_digest(Path(run) / relative) != manifest["data_files"][relative]:
        raise ValueError("Question data changed since preparation")
    rows = read_jsonl(Path(run) / relative)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows[:limit]
    return rows


def require_gate(run, manifest, phase):
    if phase == "pilot":
        return
    gate = read_json(Path(run) / "pilot_gate.json")
    if gate["config_hash"] != manifest["config_hash"] or gate["source_hash"] != source_digest():
        raise ValueError("Pilot gate is stale; rerun validation under this code/config")
    if phase == "test":
        lock = read_json(Path(run) / "analysis_lock.json")
        if lock["config_hash"] != manifest["config_hash"] or lock["source_hash"] != source_digest():
            raise ValueError("Analysis lock is stale")


def stage_fingerprint(manifest, stage):
    modules = ["engine.py"] if stage == "generate" else ["engine.py", "features.py", "answers.py"]
    if stage == "analysis":
        modules.append("analysis.py")
    return digest({"config": manifest["config_hash"], "revision": manifest["model_revision"],
                   "stage": stage, "schema": 1, "source": source_digest(modules)})


def log_failure(run, phase, stage, sample_id, error):
    write_json(Path(run) / "errors" / phase / stage / f"{sample_id}.json",
               {"id": sample_id, "error": str(error), "traceback": traceback.format_exc(),
                "source_hash": source_digest(), "time": time.time()})


def record_environment(path, current):
    if path.exists():
        previous = read_json(path)
        if previous != current:
            raise ValueError("Runtime environment changed within this stage. Preserve the run and use a new directory.")
    else:
        write_json(path, current)


def selected_rows(run, manifest, phase, limit, selected_ids):
    rows = rows_for_phase(run, manifest, phase, limit)
    records = list((Path(run) / "offload/records").glob("*.json"))
    if selected_ids is None:
        if records or (Path(run) / "offload/active.json").exists():
            raise ValueError("This remote run uses offloading. Use run-batch; analyze the restored local run.")
        return rows
    selected = set(selected_ids)
    if not selected or not selected.issubset({r["id"] for r in rows}):
        raise ValueError("Invalid batch IDs")
    exported = {sid for p in records for sid in read_json(p)["sample_ids"]}
    if selected & exported:
        raise ValueError("Refusing to regenerate packed/offloaded samples")
    return [r for r in rows if r["id"] in selected]


def generate(cfg, run, phase, limit=None, keep_going=False, selected_ids=None):
    from .engine import load_model, generate_example, environment

    run = Path(run)
    manifest = manifest_for_run(run, cfg)
    require_gate(run, manifest, phase)
    rows = selected_rows(run, manifest, phase, limit, selected_ids)
    fingerprint = stage_fingerprint(manifest, "generate")
    pending = [row for row in rows if not complete(run / "generations" / phase / row["id"], fingerprint, verify=True)]
    if not pending:
        print("All selected generations are complete.", flush=True)
        return
    model, tokenizer = load_model(cfg, manifest["model_revision"])
    if digest(tokenizer.chat_template) != manifest["chat_template_hash"]:
        raise ValueError("Chat template differs from preparation")
    record_environment(run / "generation_environment.json", environment())
    quality_ids = set(read_json(run / "sample_ids.json")["pilot"][:cfg["quality"]["generation_replay_examples"]])
    errors = 0
    for index, row in enumerate(pending, 1):
        target = run / "generations" / phase / row["id"]
        try:
            record = generate_example(model, tokenizer, row["question"], cfg, row["id"] in quality_ids)
            record.update(id=row["id"], phase=phase, fingerprint=fingerprint)
            write_json(target / "generation.json", record)
            commit_files(target, ["generation.json"], fingerprint)
            print(f"[{phase} generate {index}/{len(pending)}] {row['id']} "
                  f"tokens={record['generated_tokens']} {record['stop_reason']}", flush=True)
        except Exception as error:
            log_failure(run, phase, "generate", row["id"], error)
            errors += 1
            if not keep_going:
                raise
    if errors:
        raise RuntimeError(f"{errors} generation failures; inspect errors/ and rerun")


def label(cfg, run, phase, limit=None, selected_ids=None):
    run = Path(run)
    manifest = manifest_for_run(run, cfg)
    if file_digest(run / "data/gold.json") != manifest["data_files"]["data/gold.json"]:
        raise ValueError("Gold data changed")
    gold = read_json(run / "data/gold.json")
    fingerprint = stage_fingerprint(manifest, "generate")
    count = 0
    for row in selected_rows(run, manifest, phase, limit, selected_ids):
        folder = run / "generations" / phase / row["id"]
        if not complete(folder, fingerprint, verify=True):
            raise ValueError(f"Generation incomplete: {row['id']}")
        record = read_json(folder / "generation.json")
        result = label_answer(record["text"], gold[row["id"]], record["truncated"])
        result.update(id=row["id"], generation_checksum=file_digest(folder / "generation.json"),
                      parser_hash=source_digest(["answers.py"]))
        write_json(run / "labels" / phase / f"{row['id']}.json", result)
        count += 1
    print(f"Labeled {count} {phase} generations.", flush=True)


def extract(cfg, run, phase, limit=None, keep_going=False, recompute=False, selected_ids=None):
    import torch
    import pandas as pd
    from safetensors.torch import save_file, load_file
    from .engine import load_model, replay, environment
    from .features import project_features, check_quality, summarize_example

    run = Path(run)
    manifest = manifest_for_run(run, cfg)
    require_gate(run, manifest, phase)
    rows = selected_rows(run, manifest, phase, limit, selected_ids)
    fingerprint = stage_fingerprint(manifest, "extract")
    generation_fp = stage_fingerprint(manifest, "generate")
    pending = rows if recompute else [row for row in rows if not complete(
        run / "features" / phase / row["id"], fingerprint, verify=True)]
    if not pending:
        print("All selected feature shards are complete.", flush=True)
        return
    model, tokenizer = load_model(cfg, manifest["model_revision"])
    record_environment(run / "extraction_environment.json", environment())
    errors = 0
    for index, row in enumerate(pending, 1):
        try:
            gen_dir = run / "generations" / phase / row["id"]
            if not complete(gen_dir, generation_fp, verify=True):
                raise ValueError(f"Generation incomplete: {row['id']}")
            generation = read_json(gen_dir / "generation.json")
            ids = generation["generated_token_ids"]
            target = run / "features" / phase / row["id"]
            if recompute:
                (target / "complete.json").unlink(missing_ok=True)
            cache = run / "activation_cache" / phase / row["id"]
            cache_fp = digest({"generation": file_digest(gen_dir / "generation.json"),
                               "config": manifest["config_hash"], "hook": "block_output_pre_final_norm_v1",
                               "engine": source_digest(["engine.py"])})
            text, regions = token_regions(tokenizer, ids)
            if text != generation["text"]:
                raise ValueError("Decoded analysis text differs from saved generation")
            start = time.perf_counter()
            if model.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            if complete(cache, cache_fp, verify=True):
                tensors = load_file(str(cache / "residual.safetensors"), device="cpu")
                residual, native, positions = (tensors[key] for key in ("residual", "native_hidden", "positions"))
            else:
                reserve = cfg["extraction"]["minimum_free_disk_gib"] * 2**30
                expected = len(ids) * cfg["model"]["expected_layers"] * cfg["model"]["expected_hidden_size"]
                expected *= 2 if cfg["model"]["dtype"] == "bfloat16" else 4
                if cfg["extraction"]["cache_mode"] == "features_only":
                    expected = 8 * 1024**2
                if shutil.disk_usage(run).free < reserve + expected:
                    raise RuntimeError("Insufficient disk reserve; preserve run and free disk before resuming")
                residual, native, positions = replay(model, generation["prompt_token_ids"], ids)
                if cfg["extraction"]["cache_mode"] == "full":
                    atomic_write(cache / "residual.safetensors", lambda p: save_file(
                        {"residual": residual.contiguous(), "native_hidden": native.contiguous(),
                         "positions": positions.contiguous()}, str(p)))
                    write_json(cache / "metadata.json", {
                        "fingerprint": cache_fp, "model_revision": manifest["model_revision"],
                        "shape": list(residual.shape), "dtype": str(residual.dtype),
                        "hook": "decoder_block_output_before_final_RMSNorm", "layer_ids": list(range(1, len(residual) + 1)),
                        "target_token_ids": ids, "target_indices": list(range(len(ids))),
                        "prediction_positions": positions.tolist(),
                    })
                    commit_files(cache, ["residual.safetensors", "metadata.json"], cache_fp)
            token_frame, top_frame, diagnostic_frame, quality = project_features(model, residual, native, ids, cfg)
            quality = check_quality(quality, generation, cfg)
            quality.update(extraction_seconds=time.perf_counter() - start,
                           gpu_peak_allocated_gib=(torch.cuda.max_memory_allocated() / 2**30
                                                   if model.device.type == "cuda" else None),
                           residual_bytes=residual.numel() * residual.element_size(),
                           generated_tokens=len(ids))
            write_json(target / "quality.json", quality)
            if not quality["passed"]:
                raise RuntimeError(f"Numerical quality gate failed; inspect {target / 'quality.json'}")
            token_frame = token_frame.merge(pd.DataFrame(regions), on="token_index", validate="many_to_one")
            token_frame["prediction_position"] = token_frame.token_index + len(generation["prompt_token_ids"]) - 1
            examples, trajectory = summarize_example(token_frame, diagnostic_frame, cfg["analysis"]["trajectory_bins"])
            frames = {"token_features.parquet": token_frame, "topk_tokens.parquet": top_frame,
                      "token_diagnostics.parquet": diagnostic_frame,
                      "example_features.parquet": examples, "trajectory.parquet": trajectory}
            for filename, frame in frames.items():
                frame.insert(0, "id", row["id"])
                write_parquet(target / filename, frame)
            commit_files(target, [*frames, "quality.json"], fingerprint)
            del residual, native, token_frame, top_frame, diagnostic_frame, examples, trajectory, frames
            print(f"[{phase} extract {index}/{len(pending)}] {row['id']} "
                  f"{quality['extraction_seconds']:.1f}s", flush=True)
        except Exception as error:
            log_failure(run, phase, "extract", row["id"], error)
            errors += 1
            if not keep_going:
                raise
            if model.device.type == "cuda":
                torch.cuda.empty_cache()
    if errors:
        raise RuntimeError(f"{errors} extraction failures; inspect errors/ and rerun")


def freeze(cfg, run, analysis=False):
    run = Path(run)
    manifest = manifest_for_run(run, cfg)
    phase = "exploration" if analysis else "pilot"
    fingerprint = stage_fingerprint(manifest, "extract")
    quality_records = []
    for row in rows_for_phase(run, manifest, phase):
        target = run / "features" / phase / row["id"]
        if not complete(target, fingerprint, verify=True):
            raise ValueError(f"Complete all {phase} features before freezing: {row['id']}")
        quality = read_json(target / "quality.json")
        if not quality["passed"]:
            raise ValueError("Numerical checks failed")
        quality_records.append(quality)
        label_path = run / "labels" / phase / f"{row['id']}.json"
        if not label_path.exists():
            raise ValueError("Label every example before freezing")
        label_record = read_json(label_path)
        if (label_record.get("parser_hash") != source_digest(["answers.py"])
                or label_record["generation_checksum"] != file_digest(
                    run / "generations" / phase / row["id"] / "generation.json")):
            raise ValueError("Stale labels; run label again before freezing")
    if not analysis:
        n_checked = sum(q["generation_replay_pass"] is True for q in quality_records)
        if n_checked != cfg["quality"]["generation_replay_examples"]:
            raise ValueError("Missing generation/replay quality samples")
    else:
        require_gate(run, manifest, "exploration")
        if not complete(run / "analysis/exploration", stage_fingerprint(manifest, "analysis"), verify=True):
            raise ValueError("Finish exploration report generation before locking analysis")
        summary = read_json(run / "analysis/exploration/summary.json")
        if summary["completed"] != cfg["dataset"]["exploration_size"] or summary.get("partial"):
            raise ValueError("Analyze the full exploration split before locking test analysis")
        if summary["source_hash"] != source_digest():
            raise ValueError("Exploration analysis code changed")
    write_json(run / ("analysis_lock.json" if analysis else "pilot_gate.json"), {
        "config_hash": manifest["config_hash"], "source_hash": source_digest(),
        "checked_examples": len(quality_records), "time": time.time(),
        "max_gpu_allocated_gib": max((q["gpu_peak_allocated_gib"] or 0) for q in quality_records),
        "mean_residual_bytes": sum(q["residual_bytes"] for q in quality_records) / len(quality_records),
        "mean_extraction_seconds": sum(q["extraction_seconds"] for q in quality_records) / len(quality_records),
        "mean_generated_tokens": sum(q["generated_tokens"] for q in quality_records) / len(quality_records),
        "projected_all_phase_residual_gib": (
            sum(q["residual_bytes"] for q in quality_records) / len(quality_records)
            * sum(len(ids) for ids in read_json(run / "sample_ids.json").values()) / 2**30),
    })
    print(f"{'Analysis' if analysis else 'Pilot'} gate saved. No model weights were trained.", flush=True)
