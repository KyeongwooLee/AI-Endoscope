"""Validate input hashes, test frozen science code and detach a single recovery worker."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from common import sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    ready = json.loads((root / "READY.json").read_text())
    if sha(root / "input-bundle.zip") != ready["bundle_sha256"]:
        raise ValueError("Recovery bundle checksum")
    manifest = json.loads((root / "RUN_MANIFEST.json").read_text())
    if manifest["fingerprint"] != ready["fingerprint"]:
        raise ValueError("Recovery fingerprint")
    for name, expected in manifest["files"].items():
        if sha(root / name) != expected:
            raise ValueError("Recovery input checksum: " + name)
    status = root / "workflow_status.json"
    if status.exists():
        previous = json.loads(status.read_text())
        try:
            os.kill(previous["pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError("Refusing duplicate worker")
    environment = dict(os.environ, HF_HOME="/tmp/smollm2-recovery-cache", HF_HUB_DISABLE_XET="1",
        TMPDIR="/tmp/smollm2-recovery-tmp", CUBLAS_WORKSPACE_CONFIG=":4096:8", TOKENIZERS_PARALLELISM="false")
    subprocess.run([str(root / "venv/bin/python"), "-m", "unittest", "-v", "test_details"],
        cwd=root / "src", env=environment, check=True)
    with (root / "job.log").open("a") as stream:
        worker = subprocess.Popen([str(root / "venv/bin/python"), "-u", str(root / "src/recovery_worker.py"), "--root", str(root)],
            cwd=root, env=environment, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            close_fds=True, start_new_session=True)
    print(json.dumps(dict(pid=worker.pid, detached=True, fingerprint=ready["fingerprint"],
        persistent_output_root=str(root), cache_ephemeral=True)))


if __name__ == "__main__":
    main()
