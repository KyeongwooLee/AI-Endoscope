"""Frequent verified backup for the recovery run; ACK only after local integrity checks."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "smollm2-detailed-v1"))
from collect import verify
from common import sha, write_json

REMOTE_ROOT = "/root/ai-endoscope/smollm2-recovery-v1"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True)
    p.add_argument("--port", required=True)
    p.add_argument("--known-hosts", required=True)
    p.add_argument("--local", type=Path, required=True)
    p.add_argument("--ready", type=Path, required=True)
    p.add_argument("--once", action="store_true")
    args = p.parse_args()
    expected_fp = json.loads(args.ready.read_text())["fingerprint"]
    local = args.local.resolve()
    incoming = local / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    options = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3", "-o", "UserKnownHostsFile=" + str(Path(args.known_hosts).resolve())]
    def remote(command):
        result = subprocess.run(["ssh", *options, "-p", args.port, args.host,
            REMOTE_ROOT + "/venv/bin/python " + REMOTE_ROOT + "/src/control.py " + command],
            capture_output=True, text=True, check=True, timeout=900)
        return json.loads(result.stdout)
    def receive(name, target):
        subprocess.run(["scp", *options, "-P", args.port, args.host + ":" + name, str(target)],
            capture_output=True, check=True, timeout=1800)
    def snapshot(final=False):
        receipt = remote("snapshot")
        partial = local / "metadata.zip.partial"
        receive(receipt["path"], partial)
        if sha(partial) != receipt["sha256"] or partial.stat().st_size != receipt["bytes"]:
            raise ValueError("Snapshot archive hash")
        with zipfile.ZipFile(partial) as z:
            hashes = json.loads(z.read("snapshot_manifest.json"))
            if set(z.namelist()) != set(hashes) | {"snapshot_manifest.json"}:
                raise ValueError("Snapshot member set")
            for name, expected in hashes.items():
                if hashlib.sha256(z.read(name)).hexdigest() != expected:
                    raise ValueError("Snapshot member checksum")
        partial.replace(local / "metadata.zip")
        write_json(local / "metadata_receipt.json", receipt)
        if final:
            shutil.copyfile(local / "metadata.zip", local / "final-metadata.zip")
            write_json(local / "final_metadata_receipt.json", receipt)
    verified = {}
    last_snapshot = 0
    while True:
        try:
            status = remote("status")
            if status.get("RUN_MANIFEST", {}).get("fingerprint") != expected_fp:
                raise ValueError("Foreign recovery run")
            for receipt in status["receipts"]:
                name = receipt["name"]
                if not name.startswith("batch_") or len(name) != 10 or not name[6:].isdigit() or receipt["fingerprint"] != expected_fp:
                    raise ValueError("Unexpected recovery receipt")
                path = incoming / (name + ".zip")
                if name not in verified:
                    if not path.exists():
                        partial = path.with_suffix(".zip.partial")
                        receive(REMOTE_ROOT + "/outbox/" + name + ".zip", partial)
                        verify(partial, receipt)
                        partial.replace(path)
                    else:
                        verify(path, receipt)
                    write_json(incoming / (name + ".json"), receipt)
                    verified[name] = receipt["count"]
                    print("BACKUP_VERIFIED", name, sum(verified.values()), flush=True)
                    write_json(local / "collector_status.json", dict(verified_examples=sum(verified.values()),
                        verified_batches=len(verified), fingerprint=expected_fp, updated_unix=time.time(), complete=False))
                ack_path = incoming / (name + ".ack.json")
                if not ack_path.exists():
                    ack = remote("ack --batch " + name + " --sha256 " + receipt["archive_sha256"])
                    write_json(ack_path, ack)
            state = status.get("workflow_status", {}).get("state")
            complete = state == "complete" and len(verified) == 421 and sum(verified.values()) == 6729
            if time.time() - last_snapshot > 300 or state in ["complete", "failed"] or args.once:
                snapshot(final=complete)
                last_snapshot = time.time()
            write_json(local / "collector_status.json", dict(verified_examples=sum(verified.values()),
                verified_batches=len(verified), fingerprint=expected_fp, updated_unix=time.time(),
                remote_progress=status.get("progress"), remote_status=status.get("workflow_status"), complete=complete))
            if complete:
                print("RECOVERY_BACKUP_COMPLETE", flush=True)
                return
            if state == "failed":
                print("REMOTE_RECOVERY_FAILED", json.dumps(status["workflow_status"]), flush=True)
                raise SystemExit(2)
            if args.once:
                return
        except Exception as error:
            write_json(local / "collector_error.json", dict(error=repr(error), updated_unix=time.time()))
            print("BACKUP_RETRY", repr(error), flush=True)
            if args.once:
                raise
        time.sleep(30)


if __name__ == "__main__":
    main()
