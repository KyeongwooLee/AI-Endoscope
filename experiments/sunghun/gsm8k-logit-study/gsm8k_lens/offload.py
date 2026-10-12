"""Bounded remote batches and verified, receipt-gated transfer cleanup."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import time
from zipfile import ZipFile, ZIP_STORED

from .config import manifest_for_run
from .storage import (atomic_write, complete, digest, file_digest, read_json,
                      source_digest, write_json)


def safe_path(root, relative):
    root = Path(root).resolve()
    parts = PurePosixPath(relative).parts
    if not parts or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", p) or p in (".", "..") for p in parts):
        raise ValueError(f"Unsafe artifact path: {relative}")
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Symlink is not allowed: {path}")
    if not path.resolve().is_relative_to(root) or path.resolve() == root:
        raise ValueError("Artifact escapes run directory")
    return path


def inventory(run, phase, sample_ids):
    from .pipeline import stage_fingerprint
    run = Path(run)
    manifest = read_json(run / "manifest.json")
    files = {}
    for sid in sample_ids:
        for folder, stage in (("generations", "generate"), ("features", "extract")):
            relative = f"{folder}/{phase}/{sid}"
            target = safe_path(run, relative)
            if not complete(target, stage_fingerprint(manifest, stage), verify=True):
                raise ValueError(f"Incomplete sample: {sid}")
            marker = read_json(target / "complete.json")
            for name in [*marker["files"], "complete.json"]:
                path = safe_path(run, f"{relative}/{name}")
                files[f"{relative}/{name}"] = file_digest(path)
        relative = f"labels/{phase}/{sid}.json"
        label = read_json(safe_path(run, relative))
        if (label["generation_checksum"] != files[f"generations/{phase}/{sid}/generation.json"]
                or label["parser_hash"] != source_digest(["answers.py"])):
            raise ValueError("Stale labels cannot be exported")
        if not read_json(run / f"features/{phase}/{sid}/quality.json")["passed"]:
            raise ValueError("Numerical quality gate failed")
        files[relative] = file_digest(run / relative)
    return files


def pack_batch(cfg, run, phase, sample_ids):
    run = Path(run).resolve()
    manifest = manifest_for_run(run, cfg)
    payload = inventory(run, phase, sample_ids)
    metadata_names = ["manifest.json", "sample_ids.json", *manifest["data_files"],
                      "generation_environment.json", "extraction_environment.json"]
    files = dict(payload)
    for name in set(metadata_names):
        files[name] = file_digest(safe_path(run, name))
    batch_id = f"{phase}-{digest(sample_ids)[:16]}"
    bundle = {"schema": 1, "batch_id": batch_id, "phase": phase,
              "run_id": file_digest(run / "manifest.json"), "source_hash": source_digest(),
              "sample_ids": sample_ids, "files": files, "payload_files": sorted(payload)}
    archive = safe_path(run, f"offload/outbox/{batch_id}.zip")
    reserve = cfg["extraction"]["minimum_free_disk_gib"] * 2**30
    total = sum(safe_path(run, name).stat().st_size for name in files)
    if shutil.disk_usage(run).free < total + reserve:
        raise RuntimeError("Not enough disk for an export bundle plus reserve")

    def write_archive(path):
        with ZipFile(path, "w", compression=ZIP_STORED, allowZip64=True) as zipped:
            zipped.writestr("BUNDLE.json", json.dumps(bundle, sort_keys=True))
            for name in sorted(files):
                zipped.write(safe_path(run, name), name)
    atomic_write(archive, write_archive)
    record = {**bundle, "state": "ready", "archive": archive.relative_to(run).as_posix(),
              "sha256": file_digest(archive), "bytes": archive.stat().st_size}
    write_json(run / f"offload/records/{batch_id}.json", record)
    return public_record(run, record)


def public_record(run, record):
    return {key: record[key] for key in ("batch_id", "phase", "run_id", "sha256", "bytes")} | {
        "status": "ready", "archive": str(safe_path(run, record["archive"])),
        "examples": len(record["sample_ids"])}


def run_batch(cfg, run, phase, batch_size=128):
    from . import pipeline
    if cfg["extraction"]["cache_mode"] != "features_only":
        raise ValueError("Offloading requires features_only; raw residuals are not exported")
    if not 1 <= batch_size <= 128:
        raise ValueError("batch_size must be between 1 and 128")
    run = Path(run).resolve()
    manifest = manifest_for_run(run, cfg)
    pipeline.require_gate(run, manifest, phase)
    records = [read_json(p) for p in sorted((run / "offload/records").glob("*.json"))]
    for record in records:
        if record["run_id"] != file_digest(run / "manifest.json") or record["source_hash"] != source_digest():
            raise ValueError("Offload run identity or source code changed")
        if record["state"] == "deleting":
            raise ValueError("Interrupted cleanup: rerun ack-batch with the verified local receipt")
        if record["state"] == "ready":
            archive = safe_path(run, record["archive"])
            if not archive.exists() or file_digest(archive) != record["sha256"]:
                raise ValueError("Export archive is missing/corrupt; do not regenerate samples")
            return public_record(run, record)
    exported = {sid for record in records for sid in record["sample_ids"]}
    remaining = [r["id"] for r in pipeline.rows_for_phase(run, manifest, phase) if r["id"] not in exported]
    active = run / "offload/active.json"
    if active.exists():
        plan = read_json(active)
        if plan["phase"] != phase:
            raise ValueError("Finish the active phase batch first")
        if plan["source_hash"] != source_digest():
            raise ValueError("Code changed during an active batch")
        selected = [sid for sid in plan["sample_ids"] if sid not in exported]
    else:
        selected = remaining[:batch_size]
    if not selected:
        active.unlink(missing_ok=True)
        if remaining:
            return run_batch(cfg, run, phase, batch_size)
        return {"status": "phase_complete", "phase": phase}
    if not set(selected).issubset(remaining):
        raise ValueError("Active batch does not match prepared IDs")
    write_json(active, {"phase": phase, "sample_ids": selected, "source_hash": source_digest()})
    pipeline.generate(cfg, run, phase, selected_ids=selected)
    pipeline.label(cfg, run, phase, selected_ids=selected)
    pipeline.extract(cfg, run, phase, selected_ids=selected)
    result = pack_batch(cfg, run, phase, selected)
    active.unlink(missing_ok=True)
    return result


def stream_digest(stream):
    checksum = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        checksum.update(chunk)
    return checksum.hexdigest()


def receive_bundle(archive, expected_sha256, local_run):
    """Verify transport and every member, restore locally, then issue a receipt."""
    archive, local_run = Path(archive).resolve(), Path(local_run).resolve()
    if file_digest(archive) != expected_sha256:
        raise ValueError("Downloaded archive checksum mismatch; remote data must NOT be deleted")
    local_run.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as zipped:
        names = zipped.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members")
        if zipped.getinfo("BUNDLE.json").file_size > 4 * 1024**2:
            raise ValueError("Oversized bundle manifest")
        bundle = json.loads(zipped.read("BUNDLE.json"))
        if bundle["schema"] != 1 or set(names) != {*bundle["files"], "BUNDLE.json"}:
            raise ValueError("Unexpected bundle members/schema")
        if bundle["source_hash"] != source_digest():
            raise ValueError("Use the same experiment source version on local and remote hosts")
        if bundle["run_id"] != bundle["files"].get("manifest.json"):
            raise ValueError("Bundle run identity mismatch")
        free_needed = sum(zipped.getinfo(name).file_size for name in bundle["files"])
        if shutil.disk_usage(local_run).free < free_needed + 1024**3:
            raise RuntimeError("Insufficient local disk; remote files are unchanged")
        for name, checksum in bundle["files"].items():
            path = safe_path(local_run, name)
            with zipped.open(name) as source:
                if stream_digest(source) != checksum:
                    raise ValueError(f"Corrupt archive member: {name}")
            if path.exists() and file_digest(path) != checksum:
                raise ValueError(f"Conflicting local artifact; refusing overwrite: {name}")
        for name in bundle["files"]:
            path = safe_path(local_run, name)
            if path.exists():
                continue
            def restore(temp, name=name):
                with zipped.open(name) as source, temp.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                    target.flush()
                    os.fsync(target.fileno())
            atomic_write(path, restore)
        for name, checksum in bundle["files"].items():
            if file_digest(safe_path(local_run, name)) != checksum:
                raise ValueError("Restored local content failed verification")
    receipt = {"schema": 1, "batch_id": bundle["batch_id"], "run_id": bundle["run_id"],
               "sha256": expected_sha256, "bytes": archive.stat().st_size,
               "files_digest": digest(bundle["files"]), "verified": True,
               "local_run": str(local_run), "verified_unix": time.time()}
    path = safe_path(local_run, f"receipts/{bundle['batch_id']}.json")
    write_json(path, receipt)
    return path


def acknowledge(run, receipt_path):
    """Delete only inventoried remote payload files after a local verification receipt."""
    run = Path(run).resolve()
    receipt = read_json(receipt_path)
    record_path = safe_path(run, f"offload/records/{receipt['batch_id']}.json")
    record = read_json(record_path)
    if (receipt.get("verified") is not True or receipt.get("schema") != 1
            or receipt.get("files_digest") != digest(record["files"])
            or any(receipt.get(key) != record[key] for key in ("batch_id", "run_id", "sha256", "bytes"))
            or record["run_id"] != file_digest(run / "manifest.json")):
        raise ValueError("Invalid receipt; no remote files deleted")
    if record["state"] == "offloaded":
        return {"status": "already_offloaded", "batch_id": record["batch_id"]}
    if record["state"] not in ("ready", "deleting"):
        raise ValueError("Invalid export state")
    allowed_prefixes = tuple(f"{kind}/{record['phase']}/{sid}/" for sid in record["sample_ids"]
                             for kind in ("features", "generations"))
    allowed_labels = {f"labels/{record['phase']}/{sid}.json" for sid in record["sample_ids"]}
    paths = []
    for name in record["payload_files"]:
        if not (name.startswith(allowed_prefixes) or name in allowed_labels):
            raise ValueError("Refusing deletion outside batch payload")
        path = safe_path(run, name)
        if path.exists():
            if file_digest(path) != record["files"][name]:
                raise ValueError("Remote content changed after packing; refusing deletion")
            paths.append(path)
        elif record["state"] == "ready":
            raise ValueError("Remote payload disappeared before acknowledgment")
    archive = safe_path(run, record["archive"])
    if archive.exists() and file_digest(archive) != record["sha256"]:
        raise ValueError("Remote archive changed; refusing deletion")
    if not archive.exists() and record["state"] == "ready":
        raise ValueError("Remote archive disappeared before acknowledgment")
    # Persist the receipt and state before any unlink so interrupted cleanup is resumable.
    record.update(state="deleting", receipt=receipt)
    write_json(record_path, record)
    for path in paths:
        path.unlink()
    archive.unlink(missing_ok=True)
    parents = {safe_path(run, name).parent for name in record["payload_files"]}
    for parent in sorted(parents, key=lambda p: len(p.parts), reverse=True):
        if (parent.exists() and parent.resolve().is_relative_to(run)
                and parent != run and not any(parent.iterdir())):
            parent.rmdir()
    record["state"] = "offloaded"
    write_json(record_path, record)
    return {"status": "offloaded", "batch_id": record["batch_id"], "examples": len(record["sample_ids"])}
