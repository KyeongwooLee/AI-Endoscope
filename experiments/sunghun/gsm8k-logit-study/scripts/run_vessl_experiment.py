"""Execute the frozen protocol with verified batch offloads and durable job status."""

import argparse
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filelock import FileLock
from gsm8k_lens.storage import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--remote-project", required=True)
    parser.add_argument("--remote-run", default="/tmp/gsm8k-study/run-v2")
    parser.add_argument("--local-run", required=True)
    parser.add_argument("--ssh-bin", default="ssh")
    parser.add_argument("--scp-bin", default="scp")
    parser.add_argument("--identity")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    local_run = Path(args.local_run).resolve()
    local_run.mkdir(parents=True, exist_ok=True)
    status_path = local_run / "job_status.json"
    base = [sys.executable, str(project / "scripts/sync_from_vessl.py"),
            "--host", args.host, "--port", str(args.port),
            "--remote-project", args.remote_project, "--remote-run", args.remote_run,
            "--local-run", str(local_run), "--ssh-bin", args.ssh_bin, "--scp-bin", args.scp_bin]
    if args.identity:
        base += ["--identity", args.identity]
    status = {"state": "running", "phase": "initializing", "started_unix": time.time(),
              "host": args.host, "port": args.port, "remote_run": args.remote_run,
              "local_run": str(local_run), "summaries": {}}

    def save(phase, step):
        status.update(phase=phase, step=step, updated_unix=time.time())
        write_json(status_path, status)

    def command(argv):
        print("RUN " + " ".join(argv), flush=True)
        subprocess.run(argv, cwd=project, check=True)

    def local(command_name, phase=None):
        argv = [sys.executable, "-m", "gsm8k_lens", command_name, "--run", str(local_run)]
        if phase:
            argv += ["--phase", phase]
        command(argv)

    def send_gate(filename):
        argv = [args.scp_bin, "-P", str(args.port), "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=yes"]
        if args.identity:
            argv += ["-i", str(Path(args.identity).resolve())]
        command([*argv, str(local_run / filename), f"{args.host}:{args.remote_run}/{filename}"])

    with FileLock(str(local_run / ".job.lock"), timeout=0):
        try:
            manifest = read_json(local_run / "manifest.json")
            for phase, expected in (("pilot", manifest["config"]["dataset"]["pilot_size"]),
                                    ("exploration", manifest["config"]["dataset"]["exploration_size"]),
                                    ("test", manifest["config"]["dataset"]["expected_test_size"])):
                gate_name = {"pilot": "pilot_gate.json", "exploration": "analysis_lock.json"}.get(phase)
                summary_path = local_run / f"analysis/{phase}/summary.json"
                if gate_name and (local_run / gate_name).exists() and summary_path.exists():
                    saved = read_json(summary_path)
                    gate = read_json(local_run / gate_name)
                    if (not saved["partial"] and saved["completed"] == expected
                            and gate["config_hash"] == manifest["config_hash"]
                            and gate["source_hash"] == manifest["source_hash"]):
                        status["summaries"][phase] = saved
                        print(f"Resume: {phase} already complete and frozen", flush=True)
                        continue
                save(phase, "generate_extract_transfer")
                batches = (expected + args.batch_size - 1) // args.batch_size + 1
                command([*base, "--phase", phase, "--batch-size", str(args.batch_size), "--batches", str(batches)])
                save(phase, "analyze")
                local("analyze", phase)
                summary = read_json(local_run / f"analysis/{phase}/summary.json")
                if summary["partial"] or summary["completed"] != expected:
                    raise ValueError(f"Incomplete {phase}; next phase will not start")
                if phase == "pilot" and summary["eligible_generation_coverage"] < 0.8:
                    raise ValueError("Pilot parsing/truncation coverage is below 80%; inspect outputs before continuing")
                status["summaries"][phase] = summary
                if phase == "pilot":
                    save(phase, "freeze_validated_pilot")
                    local("freeze-pilot")
                    send_gate("pilot_gate.json")
                elif phase == "exploration":
                    # Config, primary layer, parser and source were fixed before these results.
                    save(phase, "freeze_predefined_analysis")
                    local("freeze-analysis")
                    send_gate("analysis_lock.json")
            status["state"] = "complete"
            save("test", "complete")
        except Exception as error:
            status.update(state="failed", error=str(error), traceback=traceback.format_exc())
            save(status["phase"], "failed")
            raise


if __name__ == "__main__":
    main()
