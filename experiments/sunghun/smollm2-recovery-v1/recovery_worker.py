"""Missing-only recovery using frozen original generation/extraction functions."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np
import torch
from transformers import TopPLogitsWarper

from common import sha, write_json
from extract import extract
from gsm8k_lens.answers import label_answer
from gsm8k_lens.engine import load_model
from run import generate, commit, verify_zip
from runtime_compat import cpu_cumsum_top_p, sampler_qa


def reproduce(root, model, tokenizer, config):
    report = []
    for item in json.loads((root / "input/controls.json").read_text()):
        row = item["row"]
        with np.load(root / item["npz"], allow_pickle=False) as old:
            original = json.loads(str(old["record"]))
            generated = generate(model, tokenizer, row, config)
            for key in ["prompt_token_ids", "generated_token_ids", "text", "seed"]:
                if generated[key] != original["generation"][key]:
                    raise ValueError("Reproduction token path differs: " + row["id"] + " " + key)
            arrays, qa, _ = extract(model, tokenizer, generated, config, quality=True)
            maximum, bitwise = 0., 0
            for key, value in arrays.items():
                previous = old[key]
                if np.issubdtype(value.dtype, np.floating):
                    np.testing.assert_allclose(value, previous, rtol=5e-4, atol=5e-4, equal_nan=True, err_msg=row["id"] + ":" + key)
                    finite = np.isfinite(value) & np.isfinite(previous)
                    if finite.any():
                        maximum = max(maximum, float(np.abs(value[finite] - previous[finite]).max()))
                else:
                    np.testing.assert_array_equal(value, previous)
                bitwise += int(np.array_equal(value, previous, equal_nan=True))
        report.append(dict(id=row["id"], dataset=row["dataset"], exact_token_path=True,
            feature_arrays_allclose=True, feature_array_count=len(arrays), bitwise_arrays=bitwise,
            maximum_feature_abs_difference=maximum, quality=qa))
        write_json(root / "reproduction_check.json", dict(passed=False, completed_controls=len(report), controls=report))
        print("REPRODUCTION_CONTROL_PASSED", row["id"], flush=True)
    result = dict(passed=True, controls=report, controls_excluded_from_release=True)
    write_json(root / "reproduction_check.json", result)
    return result


def work(root):
    import fcntl
    for folder in ["outbox", "acks", "pending", "results"]:
        (root / folder).mkdir(parents=True, exist_ok=True)
    lock = (root / "worker.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((root / "RUN_MANIFEST.json").read_text())
    for name, expected in manifest["files"].items():
        if sha(root / name) != expected:
            raise ValueError("Frozen input mismatch: " + name)
    fingerprint = manifest["fingerprint"]
    config = json.loads((root / "input/config.json").read_text())
    plan = json.loads((root / "input/recovery_plan.json").read_text())
    rows = json.loads((root / "input/questions.json").read_text())
    lookup = {r["id"]: r for r in rows}
    if len(rows) != len(lookup) or len(rows) != manifest["total"]:
        raise ValueError("Recovery scope mismatch")
    write_json(root / "workflow_status.json", dict(state="loading_model", pid=os.getpid(), updated_unix=time.time()))
    torch.set_num_threads(4)
    runtime_qa = sampler_qa()
    TopPLogitsWarper.__call__ = cpu_cumsum_top_p
    model, tokenizer = load_model(config, config["model"]["revision"])
    model.requires_grad_(False)
    write_json(root / "environment.json", dict(torch=torch.__version__, numpy=np.__version__,
        transformers=importlib.metadata.version("transformers"), gpu=torch.cuda.get_device_name(),
        dtype=str(next(model.parameters()).dtype), actual_parameters=sum(p.numel() for p in model.parameters()),
        chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(), sampling_runtime_qa=runtime_qa))
    if torch.__version__ != "2.3.1+cu121" or importlib.metadata.version("transformers") != "4.46.3":
        raise ValueError("Recovery must use the pinned original torch/transformers versions")
    reproduce(root, model, tokenizer, config)
    summaries, done, session_new = [], 0, 0
    domain_new = Counter()
    started = time.monotonic()
    for batch in plan["missing_batches"]:
        name = batch["batch"]
        subset = [lookup[identifier] for identifier in batch["ids"]]
        receipt_path = root / "outbox" / (name + ".json")
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt["fingerprint"] != fingerprint or receipt["ids"] != batch["ids"]:
                raise ValueError("Foreign recovery receipt")
            archive = receipt_path.with_suffix(".zip")
            if archive.exists():
                verify_zip(archive, receipt)
            else:
                ack = json.loads((root / "acks" / (name + ".json")).read_text())
                if not ack["verified_local"] or ack["archive_sha256"] != receipt["archive_sha256"]:
                    raise ValueError("Unverified missing archive")
            summaries.extend(receipt["summaries"])
            done += len(subset)
            continue
        for row in subset:
            if shutil.disk_usage(root).free < config["storage"]["minimum_free_gib"] * 2**30:
                raise RuntimeError("Persistent output disk free-space guard")
            if sum(p.stat().st_size for p in (root / "outbox").glob("*.zip")) > config["storage"]["maximum_outbox_gib"] * 2**30:
                raise RuntimeError("Outbox guard: collect backed-up batches then resume")
            pending = root / "pending" / (row["id"] + ".npz")
            if pending.exists():
                with np.load(pending, allow_pickle=False) as z:
                    previous = json.loads(str(z["record"]))
                if previous["id"] != row["id"] or previous["fingerprint"] != fingerprint:
                    raise ValueError("Foreign pending record")
                done += 1
                continue
            generation_path = pending.with_suffix(".generation.json")
            if generation_path.exists():
                saved = json.loads(generation_path.read_text())
                if saved["id"] != row["id"] or saved["fingerprint"] != fingerprint:
                    raise ValueError("Foreign pending generation")
                generation = saved["generation"]
            else:
                call_row = dict(row)
                # Only this diagnostic flag changes; seeds depend on ID, not this index.
                if domain_new[row["dataset"]] < 2:
                    call_row["dataset_index"] = 0
                generation = generate(model, tokenizer, call_row, config)
                write_json(generation_path, dict(id=row["id"], fingerprint=fingerprint, generation=generation))
            features, qa, names = extract(model, tokenizer, generation, config, quality=domain_new[row["dataset"]] < 8)
            label = label_answer(generation["text"], row["gold"], generation["truncated"])
            record = dict(**row, fingerprint=fingerprint, generation=generation, label=label, quality=qa,
                confidence_names=names, official_gsm_test=False,
                provenance=dict(kind="recovery_reinference", original_fingerprint=plan["original_fingerprint"],
                    original_batch=name, reason="original response/features not backed up before ephemeral storage reset"))
            partial = pending.with_suffix(".npz.partial")
            with partial.open("wb") as stream:
                np.savez_compressed(stream, record=np.asarray(json.dumps(record, ensure_ascii=False, allow_nan=False)), **features)
            partial.replace(pending)
            done += 1
            session_new += 1
            domain_new[row["dataset"]] += 1
            elapsed = time.monotonic() - started
            write_json(root / "progress.json", dict(completed=done, total=len(rows), session_examples=session_new,
                preserved_original=1744, combined_completed=1744 + done, combined_total=8473, last_id=row["id"],
                elapsed_session_seconds=elapsed, remaining_seconds_estimate=elapsed / session_new * (len(rows) - done),
                peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30, updated_unix=time.time()))
            print(f"RECOVERY {done}/{len(rows)} {row['id']} tokens={generation['generated_tokens']}", flush=True)
        receipt = commit(root, name, subset, fingerprint)
        summaries.extend(receipt["summaries"])
        greedy = [r for r in summaries if not r["do_sample"]]
        ratio = sum(r["quality"]["matches"] for r in greedy) / sum(r["quality"]["positions"] for r in greedy) if greedy else None
        if ratio is not None and ratio < .99:
            raise ValueError("Greedy replay QA failed")
        write_json(root / "quality_gate.json", dict(passed=True, greedy_agreement=ratio,
            independent_prefix_checks=sum(r["quality"]["independent_prefix_checks"] for r in summaries)))
        write_json(root / "workflow_status.json", dict(state="running", pid=os.getpid(), committed=len(summaries), updated_unix=time.time()))
    if len(summaries) != 6729 or len({r["id"] for r in summaries}) != 6729:
        raise ValueError("Recovery not complete")
    results = {domain: dict(answers=len(part), correct=sum(r["label"]["answer_correct"] for r in part),
        parse_failures=sum(r["label"]["parse_status"] != "ok" for r in part), truncated=sum(r["label"]["truncated"] for r in part))
        for domain in ["gsm8k", "svamp"] for part in [[r for r in summaries if r["dataset"] == domain]]}
    write_json(root / "results/labels.json", summaries)
    write_json(root / "results/summary.json", results)
    write_json(root / "job_complete.json", dict(total=6729, fingerprint=fingerprint, summary=results,
        original_responses_preserved=1744, combined_total=8473))
    write_json(root / "workflow_status.json", dict(state="complete", pid=os.getpid(), committed=6729, updated_unix=time.time()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    try:
        work(args.root.resolve())
    except Exception as error:
        write_json(args.root / "workflow_status.json", dict(state="failed", pid=os.getpid(), error=repr(error), updated_unix=time.time()))
        raise
