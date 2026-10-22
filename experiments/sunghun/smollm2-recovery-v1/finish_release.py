"""Wait for this running recovery backup, then validate/build/upload the requested release."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--recovery", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--target", required=True)
    args = p.parse_args()
    lock = (args.recovery / "release_pipeline.lock").open("a+b")
    if sys.platform == "win32":
        import msvcrt
        lock.seek(0)
        if not lock.read(1):
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    code = Path(__file__).resolve().parent.parent / "smollm2-release-v1"
    status_file = args.recovery / "release_pipeline.json"
    def state(value, **details):
        status_file.write_text(json.dumps(dict(state=value, updated_unix=time.time(), **details), indent=2), encoding="utf-8")
    try:
        while True:
            status_path = args.recovery / "collector_status.json"
            if status_path.exists():
                try:
                    status = json.loads(status_path.read_text())
                except json.JSONDecodeError:
                    time.sleep(2)
                    continue
                if status.get("complete") and (args.recovery / "final-metadata.zip").exists():
                    break
                if status.get("remote_status", {}).get("state") == "failed":
                    raise RuntimeError("Recovery failed; no release will be built or published")
                state("waiting_for_verified_backup", verified_examples=status.get("verified_examples"))
            time.sleep(60)
        state("building")
        if not (args.output / "READY_TO_PUBLISH.json").exists():
            subprocess.run([sys.executable, str(code / "build_release.py"), str(args.original), str(args.output),
                "--recovery", str(args.recovery)], check=True)
        state("publishing")
        subprocess.run([sys.executable, str(code / "publish_release.py"), str(args.output), "--target", args.target], check=True)
        state("published", url="https://github.com/KyeongwooLee/AI-Endoscope/releases/tag/smollm2-features-v1")
    except Exception as error:
        state("failed_no_publication_assumed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
