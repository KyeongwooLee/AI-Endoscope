"""Reconnectable local backup; no ACK before archive and all members are verified."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

from common import sha, write_json


def verify(path, receipt):
    if path.stat().st_size != receipt["bytes"] or sha(path) != receipt["archive_sha256"]:
        raise ValueError("Archive mismatch")
    with zipfile.ZipFile(path) as z:
        if set(z.namelist()) != set(receipt["members"]):
            raise ValueError("Unexpected member set")
        for name, expected in receipt["members"].items():
            if name.startswith(("/", "\\")) or ".." in Path(name).parts:
                raise ValueError("Unsafe ZIP member")
            if hashlib.sha256(z.read(name)).hexdigest() != expected:
                raise ValueError("Member checksum mismatch")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True)
    p.add_argument("--port", required=True)
    p.add_argument("--known-hosts", required=True)
    p.add_argument("--local", type=Path, required=True)
    p.add_argument("--once", action="store_true")
    args = p.parse_args()
    local = args.local.resolve()
    incoming = local / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    root = "/tmp/smollm2-detailed-v1"
    options = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=15",
               "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3", "-o", "UserKnownHostsFile=" + args.known_hosts]
    control = root + "/venv/bin/python " + root + "/src/control.py "
    def remote(command):
        result = subprocess.run(["ssh", *options, "-p", str(args.port), args.host, control + command],
                                capture_output=True, text=True, check=True, timeout=900)
        return json.loads(result.stdout)
    def receive(source, destination):
        subprocess.run(["scp", *options, "-P", str(args.port), args.host + ":" + source, str(destination)],
                       capture_output=True, check=True, timeout=1800)
    verified = set()
    loops = 0
    while True:
        try:
            status = remote("status")
            for receipt in status["receipts"]:
                name = receipt["name"]
                if not name.startswith("batch_") or len(name) != 10 or not name[6:].isdigit():
                    raise ValueError("Invalid remote name")
                path = incoming / (name + ".zip")
                if name not in verified:
                    if not path.exists():
                        partial = path.with_suffix(".zip.partial")
                        receive(root + "/outbox/" + name + ".zip", partial)
                        verify(partial, receipt)
                        partial.replace(path)
                    else:
                        verify(path, receipt)
                    write_json(incoming / (name + ".json"), receipt)
                    verified.add(name)
                    print("BACKUP_VERIFIED", name, receipt["count"], flush=True)
                if not (incoming / (name + ".ack.json")).exists():
                    result = remote("ack --batch " + name + " --sha256 " + receipt["archive_sha256"])
                    write_json(incoming / (name + ".ack.json"), result)
            terminal = status.get("workflow_status", {}).get("state") in ["complete", "failed"]
            if loops % 5 == 0 or args.once or terminal:
                info = remote("snapshot")
                partial = local / "metadata.zip.partial"
                receive(info["path"], partial)
                if sha(partial) != info["sha256"] or partial.stat().st_size != info["bytes"]:
                    raise ValueError("Metadata ZIP mismatch")
                with zipfile.ZipFile(partial) as z:
                    hashes = json.loads(z.read("snapshot_manifest.json"))
                    assert set(z.namelist()) == set(hashes) | {"snapshot_manifest.json"}
                    for name, expected in hashes.items():
                        assert hashlib.sha256(z.read(name)).hexdigest() == expected
                partial.replace(local / "metadata.zip")
                write_json(local / "metadata_receipt.json", info)
            count = sum(json.loads(p.read_text())["count"] for p in incoming.glob("batch_*.json") if not p.name.endswith(".ack.json"))
            write_json(local / "collector_status.json", dict(verified_examples=count, verified_batches=len(verified),
                updated_unix=time.time(), remote_progress=status.get("progress"), remote_status=status.get("workflow_status")))
            if args.once or terminal:
                return
        except Exception as error:
            write_json(local / "collector_error.json", dict(error=repr(error), updated_unix=time.time()))
            print("BACKUP_RETRY", repr(error), flush=True)
            if args.once:
                raise
        loops += 1
        time.sleep(60)


if __name__ == "__main__":
    main()
