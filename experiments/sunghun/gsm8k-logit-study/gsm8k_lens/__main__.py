import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from .config import load_config


def main():
    if sys.platform == "linux":
        scratch = Path(os.environ.get("GSM8K_SCRATCH", "/tmp/gsm8k-study"))
        os.environ.setdefault("HF_HOME", str(scratch / "hf-cache"))
        os.environ.setdefault("XDG_CACHE_HOME", str(scratch / "xdg-cache"))
        os.environ.setdefault("TMPDIR", str(scratch / "tmp"))
        Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    parser = argparse.ArgumentParser(description="GSM8K self-generated logit-lens study")
    parser.add_argument("command", choices=["doctor", "prepare", "generate", "label", "extract",
                                           "analyze", "freeze-pilot", "freeze-analysis",
                                           "run-batch", "receive-batch", "ack-batch"])
    parser.add_argument("--config", default="configs/rtx3090.json")
    default_run = scratch / "run-v2" if sys.platform == "linux" else Path(tempfile.gettempdir()) / "gsm8k-study/run-v2"
    parser.add_argument("--run", default=os.environ.get("RUN_DIR", str(default_run)))
    parser.add_argument("--phase", choices=["pilot", "exploration", "test"], default="pilot")
    parser.add_argument("--limit", type=int, help="First N selected IDs, including already completed IDs")
    parser.add_argument("--keep-going", action="store_true", help="Record per-example errors; exit nonzero if any fail")
    parser.add_argument("--allow-partial", action="store_true", help="Diagnostic analysis of available examples")
    parser.add_argument("--recompute-features", action="store_true", help="Replace feature shards, reusing compatible residual caches")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--archive", help="Downloaded local ZIP for receive-batch")
    parser.add_argument("--sha256", help="Expected archive checksum received over SSH")
    parser.add_argument("--receipt", help="Verified local receipt for ack-batch")
    parser.add_argument("--expected-run-id", help="Refuse a missing/replaced remote run manifest")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.limit is not None and args.command not in ("generate", "label", "extract"):
        parser.error("--limit is only valid for generate, label, extract")
    if args.allow_partial and args.command != "analyze":
        parser.error("--allow-partial is only valid for analyze")
    if args.recompute_features and args.command != "extract":
        parser.error("--recompute-features is only valid for extract")
    if args.command == "doctor":
        import shutil
        from .engine import configure_runtime, environment
        configure_runtime(cfg)
        info = environment()
        root = Path(args.run).resolve()
        while not root.exists():
            root = root.parent
        info.update(free_disk_gib=shutil.disk_usage(root).free / 2**30,
                    storage_path=str(root),
                    cache_mode=cfg["extraction"]["cache_mode"],
                    projection_chunk_tokens=cfg["extraction"]["projection_chunk_tokens"])
        if hasattr(os, "statvfs"):
            fs = os.statvfs(root)
            info["free_inodes"] = fs.f_favail
        print(json.dumps(info, indent=2))
        return
    from filelock import FileLock
    from . import pipeline
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    with FileLock(str(run / ".writer.lock"), timeout=0):
        if args.command == "prepare":
            pipeline.prepare(cfg, run)
        elif args.command == "generate":
            pipeline.generate(cfg, run, args.phase, args.limit, args.keep_going)
        elif args.command == "extract":
            pipeline.extract(cfg, run, args.phase, args.limit, args.keep_going, args.recompute_features)
        elif args.command == "label":
            pipeline.label(cfg, run, args.phase, args.limit)
        elif args.command == "analyze":
            from .analysis import analyze
            analyze(cfg, run, args.phase, args.allow_partial)
        elif args.command in ("freeze-pilot", "freeze-analysis"):
            pipeline.freeze(cfg, run, analysis=args.command == "freeze-analysis")
        elif args.command == "run-batch":
            from .offload import run_batch
            from .storage import file_digest
            if args.expected_run_id and (not (run / "manifest.json").exists()
                    or file_digest(run / "manifest.json") != args.expected_run_id):
                raise ValueError("Remote run state is missing/replaced. Restore its metadata/transfer ledger before resuming.")
            pipeline.prepare(cfg, run)
            result = run_batch(cfg, run, args.phase, args.batch_size)
            print("OFFLOAD_RESULT " + json.dumps(result), flush=True)
        elif args.command == "receive-batch":
            from .offload import receive_bundle
            if not args.archive or not args.sha256:
                parser.error("receive-batch requires --archive and --sha256")
            receipt = receive_bundle(args.archive, args.sha256, run)
            print(json.dumps({"receipt": str(receipt)}))
        elif args.command == "ack-batch":
            from .offload import acknowledge
            if not args.receipt:
                parser.error("ack-batch requires --receipt")
            print("OFFLOAD_RESULT " + json.dumps(acknowledge(run, args.receipt)), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
