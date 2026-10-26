"""Restore verified recovery receipts on a fresh host without regenerating backed-up IDs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "smollm2-release-v1"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "smollm2-detailed-v1"))
from build_release import RECOVERY_FINGERPRINT, encode, require
from collect import verify
from common import sha


def resume_state(prepared, backup):
    ready = json.loads((prepared / "READY.json").read_text())
    bundle = prepared / "input-bundle.zip"
    require(sha(bundle) == ready["bundle_sha256"], "Frozen input bundle changed")
    require(ready["fingerprint"] == RECOVERY_FINGERPRINT, "Wrong recovery run")
    with zipfile.ZipFile(bundle) as z:
        manifest = json.loads(z.read("RUN_MANIFEST.json"))
        require(manifest["fingerprint"] == RECOVERY_FINGERPRINT, "Manifest fingerprint")
        require(hashlib.sha256(json.dumps(manifest["files"], sort_keys=True).encode()).hexdigest() == RECOVERY_FINGERPRINT,
            "Frozen file table fingerprint")
        for name, expected in manifest["files"].items():
            require(hashlib.sha256(z.read(name)).hexdigest() == expected, "Frozen member changed: " + name)
        plan = json.loads(z.read("input/recovery_plan.json"))
    expected = {b["batch"]: b["ids"] for b in plan["missing_batches"]}
    incoming = backup / "incoming"
    files, verified, seen = {}, [], set()
    archives = sorted(incoming.glob("batch_[0-9][0-9][0-9][0-9].zip"))
    for archive in archives:
        name = archive.stem
        receipt_path = incoming / (name + ".json")
        require(receipt_path.exists(), "Archive without receipt: " + name)
        receipt = json.loads(receipt_path.read_text())
        require(name in expected and receipt["ids"] == expected[name], "Receipt ID coverage: " + name)
        require(receipt["name"] == name and receipt["count"] == len(receipt["ids"]), "Receipt count/name")
        require(receipt["fingerprint"] == RECOVERY_FINGERPRINT, "Foreign recovery receipt")
        require(not seen.intersection(receipt["ids"]), "Duplicate recovery IDs")
        verify(archive, receipt)
        with zipfile.ZipFile(archive) as z:
            records = [json.loads(line) for line in z.read("generations.jsonl").splitlines()]
        require([r["id"] for r in records] == receipt["ids"], "Generation IDs differ")
        require(all(r["fingerprint"] == RECOVERY_FINGERPRINT for r in records), "Foreign generation fingerprint")
        require([r["id"] for r in receipt["summaries"]] == receipt["ids"], "Summary IDs differ")
        require(all(r["label"] == s["label"] for r, s in zip(records, receipt["summaries"])), "Summary labels differ")
        seen.update(receipt["ids"])
        files["outbox/" + name + ".json"] = receipt_path.read_bytes()
        # This ACK certifies the archive is still verified on this local machine.
        files["acks/" + name + ".json"] = encode(dict(verified_local=True,
            archive_sha256=receipt["archive_sha256"], bytes=receipt["bytes"], restored_from_verified_local=True))
        verified.append(dict(batch=name, count=receipt["count"], sha256=receipt["archive_sha256"], ids=receipt["ids"]))
    audit = dict(created_utc=datetime.now(timezone.utc).isoformat(), fingerprint=RECOVERY_FINGERPRINT,
        reason="Fresh VESSL host has no prior study; resume immutable run from locally verified batch receipts",
        verified_batches=len(verified), verified_recovery_responses=len(seen), remaining_recovery_responses=6729-len(seen),
        preserved_original_responses=1744, combined_verified_responses=1744+len(seen),
        source_bundle_sha256=ready["bundle_sha256"], no_scientific_source_or_input_changed=True,
        archive_and_all_member_sha256_verified=True, verified=verified)
    files["resume_audit.json"] = encode(audit)
    files["progress.json"] = encode(dict(completed=len(seen), total=6729, preserved_original=1744,
        combined_completed=1744+len(seen), combined_total=8473, session_examples=0, resume_initialized=True))
    return files, audit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--backup", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    require(not args.output.exists(), "Do not overwrite a previous resume package")
    files, audit = resume_state(args.prepared.resolve(), args.backup.resolve())
    args.output.mkdir(parents=True)
    archive = args.output / "resume-state.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    (args.output / "resume_audit.json").write_bytes(encode(audit))
    summary = {k: v for k, v in audit.items() if k != "verified"}
    summary["resume_state_sha256"] = sha(archive)
    (args.output / "READY.json").write_bytes(encode(summary))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
