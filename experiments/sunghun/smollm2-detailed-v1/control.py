"""Remote inventory, bounded metadata snapshot, and verified-ACK ZIP cleanup."""
import argparse
import json
import os
from pathlib import Path
import re
import zipfile

from common import sha, write_json
from run import verify_zip


def acknowledge(root, name, checksum):
    if not re.fullmatch(r"batch_[0-9]{4}", name):
        raise ValueError("Invalid batch name")
    receipt = json.loads((root / "outbox" / (name + ".json")).read_text())
    if checksum != receipt["archive_sha256"]:
        raise ValueError("ACK checksum mismatch")
    archive = root / "outbox" / (name + ".zip")
    if archive.exists():
        verify_zip(archive, receipt)
    # Persist evidence before deleting only the named, locally verified transport ZIP.
    write_json(root / "acks" / (name + ".json"), dict(verified_local=True, archive_sha256=checksum, bytes=receipt["bytes"]))
    archive.unlink(missing_ok=True)
    return dict(acknowledged=name, archive_removed=True)


def snapshot(root):
    members = []
    for folder in ["src", "input", "results"]:
        members += [p for p in (root / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    members += [p for p in root.glob("*.json") if p.is_file()]
    members += list((root / "outbox").glob("*.json")) + list((root / "acks").glob("*.json"))
    hashes = {p.relative_to(root).as_posix(): sha(p) for p in members}
    target = root / "metadata.zip"
    partial = target.with_suffix(".zip.partial")
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as z:
        for p in members:
            z.write(p, p.relative_to(root).as_posix())
        z.writestr("snapshot_manifest.json", json.dumps(hashes))
    partial.replace(target)
    return dict(path=str(target), sha256=sha(target), bytes=target.stat().st_size)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["status", "snapshot", "ack"])
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument("--batch")
    p.add_argument("--sha256")
    args = p.parse_args()
    root = args.root.resolve()
    if args.action == "ack":
        result = acknowledge(root, args.batch, args.sha256)
    elif args.action == "snapshot":
        result = snapshot(root)
    else:
        result = dict(receipts=[json.loads(p.read_text()) for p in sorted((root / "outbox").glob("batch_*.json"))])
        for name in ["progress", "quality_gate", "job_complete", "workflow_status", "RUN_MANIFEST"]:
            path = root / (name + ".json")
            if path.exists():
                result[name] = json.loads(path.read_text())
        pid = result.get("workflow_status", {}).get("pid")
        try:
            os.kill(pid, 0)
            result["pid_exists"] = True
        except (ProcessLookupError, TypeError):
            result["pid_exists"] = False
    print(json.dumps(result))


if __name__ == "__main__":
    main()
