"""Validate a prepared bundle and start the worker independent of SSH."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile


def main():
    root = Path("/tmp/smollm2-detailed-v1")
    ready = json.loads((root / "READY.json").read_text())
    assert hashlib.sha256((root / "input-bundle.zip").read_bytes()).hexdigest() == ready["bundle_sha256"]
    if (root / "workflow_status.json").exists():
        previous = json.loads((root / "workflow_status.json").read_text())
        try:
            os.kill(previous["pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError("Refusing duplicate worker; inspect PID first")
    if (root / "RUN_MANIFEST.json").exists():
        assert json.loads((root / "RUN_MANIFEST.json").read_text())["fingerprint"] == ready["fingerprint"]
    with zipfile.ZipFile(root / "input-bundle.zip") as z:
        assert all((root / name).resolve().is_relative_to(root) for name in z.namelist())
        z.extractall(root)
    environment = dict(os.environ, HF_HOME=str(root / "cache"), HF_HUB_DISABLE_XET="1",
        TMPDIR=str(root / "tmp"), CUBLAS_WORKSPACE_CONFIG=":4096:8", TOKENIZERS_PARALLELISM="false")
    subprocess.run([str(root / "venv/bin/python"), "-m", "unittest", "-v", "test_details"],
                   cwd=root / "src", env=environment, check=True)
    with (root / "job.log").open("a") as stream:
        child = subprocess.Popen([str(root / "venv/bin/python"), "-u", str(root / "src/run.py"), "--root", str(root)],
            cwd=root, env=environment, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            close_fds=True, start_new_session=True)
    print(json.dumps(dict(pid=child.pid, detached=True, fingerprint=ready["fingerprint"])))


if __name__ == "__main__":
    main()
