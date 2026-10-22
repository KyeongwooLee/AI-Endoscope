"""Detached, bounded-storage generation and detailed extraction with resumable batches."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import time
import zipfile

import numpy as np
import torch
from transformers import StoppingCriteriaList

from common import sha, write_json
from extract import extract, TOKEN_METRICS, TRAJECTORY_METRICS, CHECKPOINT_METRICS
from gsm8k_lens.answers import label_answer
from gsm8k_lens.engine import load_model, make_generation_config, FinalAnswerLineStop


@torch.inference_mode()
def generate(model, tokenizer, row, config):
    seed = int(hashlib.sha256(f"{config['seed']}:{row['id']}".encode()).hexdigest()[:8], 16) % (2**31)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    messages = [{"role": "system", "content": config["prompt"]["system"]},
                {"role": "user", "content": config["prompt"]["user_template"].replace("{question}", row["question"])}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if len(prompt) > config["generation"]["max_prompt_tokens"]:
        raise ValueError("Refusing silent prompt truncation")
    gcfg = make_generation_config(model, tokenizer, config)
    decoding = config["datasets"][row["dataset"]]
    gcfg.do_sample = decoding["do_sample"]
    if gcfg.do_sample:
        gcfg.temperature, gcfg.top_p, gcfg.top_k = decoding["temperature"], decoding["top_p"], decoding["top_k"]
    stop = FinalAnswerLineStop(tokenizer, len(prompt))
    ids = torch.tensor([prompt], device=model.device)
    quality = row["dataset_index"] < config["quality"]["generation_logit_examples"]
    started = time.monotonic()
    output = model.generate(ids, attention_mask=torch.ones_like(ids), generation_config=gcfg,
        stopping_criteria=StoppingCriteriaList([stop]), return_dict_in_generate=True, output_logits=quality)
    generated = output.sequences[0, len(prompt):].tolist()
    eos = gcfg.eos_token_id if isinstance(gcfg.eos_token_id, list) else [gcfg.eos_token_id]
    terminated = bool(generated and generated[-1] in eos)
    record = dict(prompt_token_ids=prompt, generated_token_ids=generated,
        text=tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False),
        seed=seed, prompt_tokens=len(prompt), generated_tokens=len(generated),
        truncated=not (terminated or stop.finished),
        stop_reason="eos" if terminated else "final_answer_line" if stop.finished else "max_new_tokens",
        effective_generation_config=gcfg.to_dict(), generation_seconds=time.monotonic() - started)
    if quality:
        record["generation_quality"] = {"generated_logprob": [float(lp[0].float().log_softmax(-1)[token])
                    for lp, token in zip(output.logits, generated)]}
    return record


def verify_zip(path, receipt):
    assert sha(path) == receipt["archive_sha256"] and path.stat().st_size == receipt["bytes"]
    with zipfile.ZipFile(path) as z:
        assert set(z.namelist()) == set(receipt["members"])
        for name, expected in receipt["members"].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected


def commit(root, name, rows, fingerprint):
    contents, records = {}, []
    for row in rows:
        path = root / "pending" / (row["id"] + ".npz")
        with np.load(path, allow_pickle=False) as data:
            record = json.loads(str(data["record"]))
        assert record["fingerprint"] == fingerprint and record["id"] == row["id"]
        records.append(record)
        contents["features/" + row["id"] + ".npz"] = path.read_bytes()
    contents["generations.jsonl"] = "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in records).encode()
    path = root / "outbox" / (name + ".zip")
    partial = path.with_suffix(".zip.partial")
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_STORED) as z:
        for member, data in contents.items():
            z.writestr(member, data)
    partial.replace(path)
    summaries = [dict(id=r["id"], dataset=r["dataset"], label=r["label"], quality=r["quality"],
                      do_sample=r["generation"]["effective_generation_config"]["do_sample"],
                      generated_tokens=r["generation"]["generated_tokens"]) for r in records]
    receipt = dict(name=name, ids=[r["id"] for r in rows], count=len(rows), fingerprint=fingerprint,
        archive_sha256=sha(path), bytes=path.stat().st_size, summaries=summaries,
        members={k: hashlib.sha256(v).hexdigest() for k, v in contents.items()})
    verify_zip(path, receipt)
    write_json(root / "outbox" / (name + ".json"), receipt)
    for row in rows:
        for suffix in [".npz", ".generation.json"]:
            (root / "pending" / (row["id"] + suffix)).unlink(missing_ok=True)
    return receipt


def quality_gate(root, config, summaries, fingerprint):
    count = config["quality"]["pilot_examples"]
    if len(summaries) < count:
        return
    pilot = summaries[:count]
    greedy = [r for r in pilot if not r["do_sample"]]
    agreement = sum(r["quality"]["matches"] for r in greedy) / sum(r["quality"]["positions"] for r in greedy)
    if agreement < config["quality"]["minimum_greedy_checkpoint_agreement"]:
        raise ValueError("Greedy generation/replay gate failed: " + str(agreement))
    if not sum(r["quality"]["independent_prefix_checks"] for r in greedy):
        raise ValueError("Missing independent prefix QA")
    write_json(root / "quality_gate.json", dict(passed=True, examples=count, greedy_agreement=agreement,
        parse_failures=sum(r["label"]["parse_status"] != "ok" for r in pilot), fingerprint=fingerprint))


def main():
    import fcntl
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    for name in ["outbox", "pending", "acks", "results"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    lock = (root / "worker.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write_json(root / "workflow_status.json", dict(state="starting", pid=os.getpid(), updated_unix=time.time()))
    config = json.loads((root / "input/config.json").read_text())
    manifest = json.loads((root / "RUN_MANIFEST.json").read_text())
    for relative, expected in manifest["files"].items():
        assert sha(root / relative) == expected, relative
    fingerprint = manifest["fingerprint"]
    rows = json.loads((root / "input/questions.json").read_text())
    assert len(rows) == len({r["id"] for r in rows}) == manifest["total"] == 8473
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    model, tokenizer = load_model(config, config["model"]["revision"])
    model.requires_grad_(False)
    write_json(root / "environment.json", dict(torch=torch.__version__, numpy=np.__version__,
        transformers=importlib.metadata.version("transformers"), gpu=torch.cuda.get_device_name(),
        dtype=str(next(model.parameters()).dtype), actual_parameters=sum(p.numel() for p in model.parameters()),
        chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()))
    write_json(root / "feature_schema.json", dict(
        token_metrics=TOKEN_METRICS, trajectory_metrics=TRAJECTORY_METRICS, checkpoint_metrics=CHECKPOINT_METRICS,
        checkpoint_hidden_axes=["available t", "embedding+blocks0..24", "hidden2048"],
        trajectory_axes=["prediction token t", "blocks1..24", "metric"],
        pooled_hidden_axes=["blocks1..24", "end,whole_mean,number_mean", "hidden2048"],
        prefix_readout_axes=["prediction token0..128", "configured readout layer", "token_metric"],
        checkpoint_readout_axes=["available t", "blocks1..24", "checkpoint_metric"],
        mask_columns=["non_special", "whole_nonwhitespace", "final_number_nonwhitespace"],
        region_codes={"excluded": 0, "reasoning": 1, "final_number": 2, "format": 3, "unparsed": 4},
        offset_unit="Python Unicode codepoint half-open range; -1 for specials",
        raw_hidden="FP32, before final RMSNorm; t=0 canonical prompt-only",
        full_vocabulary_logits_saved=False, full_token_dense_residuals_saved=False))
    batch_size = config["storage"]["batch_examples"]
    summaries = []
    start_time, new = time.monotonic(), 0
    for start in range(0, len(rows), batch_size):
        name = f"batch_{start // batch_size:04d}"
        subset = rows[start:start + batch_size]
        receipt_path = root / "outbox" / (name + ".json")
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            assert receipt["fingerprint"] == fingerprint and receipt["ids"] == [r["id"] for r in subset]
            archive = root / "outbox" / (name + ".zip")
            if archive.exists():
                verify_zip(archive, receipt)
            else:
                ack = json.loads((root / "acks" / (name + ".json")).read_text())
                assert ack["archive_sha256"] == receipt["archive_sha256"] and ack["verified_local"]
            summaries.extend(receipt["summaries"])
            quality_gate(root, config, summaries, fingerprint)
            continue
        for row in subset:
            storage = config["storage"]
            if shutil.disk_usage(root).free < storage["minimum_free_gib"] * 2**30:
                raise RuntimeError("Disk guard: preserve committed and pending files")
            if sum(p.stat().st_size for p in (root / "outbox").glob("*.zip")) > storage["maximum_outbox_gib"] * 2**30:
                raise RuntimeError("Unbacked outbox reached cap; collect and resume")
            if new % 16 == 0 and sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink()) > storage["maximum_study_gib"] * 2**30:
                raise RuntimeError("Study exceeds 35 GiB guard; preserve files")
            pending = root / "pending" / (row["id"] + ".npz")
            if pending.exists():
                with np.load(pending, allow_pickle=False) as data:
                    record = json.loads(str(data["record"]))
                assert record["fingerprint"] == fingerprint and record["id"] == row["id"]
                continue
            gpath = pending.with_suffix(".generation.json")
            if gpath.exists():
                saved = json.loads(gpath.read_text())
                assert saved["fingerprint"] == fingerprint and saved["id"] == row["id"]
                generation = saved["generation"]
            else:
                generation = generate(model, tokenizer, row, config)
                write_json(gpath, dict(id=row["id"], fingerprint=fingerprint, generation=generation))
            features, qa, confidence_names = extract(model, tokenizer, generation, config,
                quality=row["dataset_index"] < config["quality"]["prefix_checks"])
            label = label_answer(generation["text"], row["gold"], generation["truncated"])
            record = dict(**row, fingerprint=fingerprint, generation=generation, label=label, quality=qa,
                          confidence_names=confidence_names, official_gsm_test=False)
            partial = pending.with_suffix(".npz.partial")
            with partial.open("wb") as stream:
                np.savez_compressed(stream, record=np.asarray(json.dumps(record, ensure_ascii=False, allow_nan=False)), **features)
            partial.replace(pending)
            new += 1
            done = start + subset.index(row) + 1
            elapsed = time.monotonic() - start_time
            write_json(root / "progress.json", dict(completed=done, total=len(rows), committed=start,
                last_id=row["id"], session_examples=new, elapsed_session_seconds=elapsed,
                remaining_seconds_estimate=elapsed / new * (len(rows) - done),
                peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30, updated_unix=time.time()))
            print(f"{done}/{len(rows)} {row['id']} tokens={generation['generated_tokens']} GPU={torch.cuda.max_memory_allocated()/2**30:.2f}GiB", flush=True)
        receipt = commit(root, name, subset, fingerprint)
        summaries.extend(receipt["summaries"])
        quality_gate(root, config, summaries, fingerprint)
        write_json(root / "workflow_status.json", dict(state="running", pid=os.getpid(), committed=len(summaries), updated_unix=time.time()))
    result = {dataset: dict(answers=len(part), correct=sum(r["label"]["answer_correct"] for r in part),
                parse_failures=sum(r["label"]["parse_status"] != "ok" for r in part), truncated=sum(r["label"]["truncated"] for r in part))
              for dataset in ["gsm8k", "svamp"] for part in [[r for r in summaries if r["dataset"] == dataset]]}
    write_json(root / "results/labels.json", summaries)
    write_json(root / "results/summary.json", result)
    write_json(root / "job_complete.json", dict(total=len(summaries), summary=result, fingerprint=fingerprint))
    write_json(root / "workflow_status.json", dict(state="complete", pid=os.getpid(), committed=len(summaries), updated_unix=time.time()))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        import sys
        import traceback
        traceback.print_exc()
        if "--root" in sys.argv:
            root = Path(sys.argv[sys.argv.index("--root") + 1])
            write_json(root / "workflow_status.json", dict(state="failed", pid=os.getpid(), error=repr(error), updated_unix=time.time()))
        raise
