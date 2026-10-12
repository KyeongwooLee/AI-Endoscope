"""Run bounded remote batches; pull, verify, then acknowledge deletion. No auto-freeze."""

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filelock import FileLock
from gsm8k_lens.offload import receive_bundle
from gsm8k_lens.storage import file_digest, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="SSH config alias or user@hostname")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--identity", help="SSH private-key path; never copied to the remote host")
    parser.add_argument("--ssh-bin", default="ssh", help="SSH executable path")
    parser.add_argument("--scp-bin", default="scp", help="SCP executable path")
    parser.add_argument("--remote-project", required=True, help="Absolute Linux project path")
    parser.add_argument("--remote-run", default="/tmp/gsm8k-study/run-v2")
    parser.add_argument("--remote-python", help="Default: REMOTE_PROJECT/.venv/bin/python")
    parser.add_argument("--local-run", required=True)
    parser.add_argument("--phase", choices=["pilot", "exploration", "test"], default="pilot")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batches", type=int, default=1, help="Maximum batches this invocation; never advances phase")
    parser.add_argument("--queue-only", action="store_true", help="Collect detached producer outputs; never generate")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*", args.host):
        parser.error("Use a simple SSH alias or user@hostname")
    if not 1 <= args.port <= 65535 or not 1 <= args.batch_size <= 128 or args.batches < 1:
        parser.error("Invalid port or batch count/size")
    remote_project = PurePosixPath(args.remote_project)
    remote_run = PurePosixPath(args.remote_run)
    remote_python = args.remote_python or str(remote_project / ".venv/bin/python")
    for path in (str(remote_project), str(remote_run), remote_python):
        if not path.startswith("/") or not re.fullmatch(r"/[A-Za-z0-9_./-]+", path) or ".." in PurePosixPath(path).parts:
            parser.error("Remote paths must be absolute Linux paths without spaces or '..'")
    common = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
              "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6",
              "-o", "ConnectTimeout=20"]
    if args.identity:
        common += ["-i", str(Path(args.identity).resolve())]
    ssh = [args.ssh_bin, "-p", str(args.port), *common, args.host]
    scp = [args.scp_bin, "-P", str(args.port), *common]
    local_run = Path(args.local_run).resolve()
    incoming = local_run / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)

    def remote(command, *extra):
        argv = [remote_python, "-m", "gsm8k_lens", command,
                "--config", str(remote_project / "configs/rtx3090.json"),
                "--run", str(remote_run), *extra]
        if args.queue_only:
            if not (local_run / 'manifest.json').exists():
                raise ValueError('Queue collection requires an existing manifest')
            argv = [remote_python, str(remote_project / 'scripts/remote_queue.py'),
                    '--action', 'list' if command == 'run-batch' else 'ack',
                    '--config', str(remote_project / 'configs/rtx3090.json'),
                    '--run', str(remote_run), '--expected-run-id', file_digest(local_run / 'manifest.json')]
            if command == 'ack-batch':
                argv += list(extra)
            else:
                argv += ['--phase', args.phase]
        shell_command = "cd " + shlex.quote(str(remote_project)) + " && " + shlex.join(argv)
        result = None
        with subprocess.Popen([*ssh, shell_command], stdout=subprocess.PIPE, text=True) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                if line.startswith("OFFLOAD_RESULT "):
                    result = json.loads(line[len("OFFLOAD_RESULT "):])
            if process.wait() != 0:
                raise RuntimeError("SSH operation failed; no automatic deletion retry")
        if result is None:
            raise RuntimeError("Remote command did not return an offload result")
        return result

    def acknowledge_local_receipt(receipt):
        remote_receipt = str(remote_run / "offload" / (receipt.stem + ".receipt.json"))
        subprocess.run([*scp, str(receipt), f"{args.host}:{remote_receipt}"], check=True)
        return remote("ack-batch", "--receipt", remote_receipt)

    with FileLock(str(local_run / ".writer.lock"), timeout=0):
        # A lost SSH response may leave deletion half-finished. Receipts are durable locally.
        for receipt in sorted((local_run / "receipts").glob("*.json")):
            if not (incoming / (receipt.stem + ".acked.json")).exists():
                saved = read_json(receipt)
                receive_bundle(incoming / f"{receipt.stem}.zip", saved["sha256"], local_run)
                acknowledge_local_receipt(receipt)
                write_json(incoming / (receipt.stem + ".acked.json"), {"acknowledged": True})
        for _ in range(args.batches):
            expected = (["--expected-run-id", file_digest(local_run / "manifest.json")]
                        if (local_run / "manifest.json").exists() else [])
            result = remote("run-batch", "--phase", args.phase, "--batch-size", str(args.batch_size), *expected)
            if result['status'] == 'waiting':
                print('Producer running; no queued bundle yet.', flush=True)
                break
            if result["status"] == "phase_complete":
                print("Phase transferred. Analyze locally and review before freezing the next gate.")
                break
            if result["status"] != "ready":
                raise RuntimeError("Unexpected remote batch status")
            batch_id = result["batch_id"]
            if not re.fullmatch(r"(?:pilot|exploration|test)-[0-9a-f]{16}", batch_id):
                raise ValueError("Invalid remote batch ID")
            remote_archive = remote_run / "offload/outbox" / f"{batch_id}.zip"
            if result["archive"] != str(remote_archive):
                raise ValueError("Unexpected remote archive path")
            archive = incoming / f"{batch_id}.zip"
            if not archive.exists() or file_digest(archive) != result["sha256"]:
                partial = incoming / f"{batch_id}.zip.part"
                subprocess.run([*scp, f"{args.host}:{remote_archive}", str(partial)], check=True)
                if partial.stat().st_size != result["bytes"] or file_digest(partial) != result["sha256"]:
                    raise ValueError("Downloaded bytes/checksum differ; remote files are retained")
                partial.replace(archive)
            receipt = receive_bundle(archive, result["sha256"], local_run)
            acknowledge_local_receipt(receipt)
            write_json(incoming / f"{batch_id}.acked.json", {"acknowledged": True})
            print(f"Verified locally and removed remote payload: {batch_id}", flush=True)


if __name__ == "__main__":
    main()
