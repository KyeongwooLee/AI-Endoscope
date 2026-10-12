"""CPU-only loader for the train feature release. Never enables NumPy pickle."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np

CHECKPOINTS = (0, 32, 64, 128)
RAW_LAYERS = (7, 10, 13, 14)
READOUT_LAYERS = (7, 10, 14)
READOUT_METRICS = ("entropy", "margin", "top1_probability")
STATS = ("current", "mean", "std", "change_from_t0")
SCALAR_NAMES = tuple(f"l{layer}.{metric}.{stat}" for layer in READOUT_LAYERS
                     for metric in READOUT_METRICS for stat in STATS)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            value.update(chunk)
    return value.hexdigest()


def verify_checksums(root):
    root = Path(root).resolve()
    count = 0
    for line in (root / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        relative = PurePosixPath(name)
        path = (root / name).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root):
            raise ValueError(f"Unsafe checksum path: {name}")
        if sha256(path) != expected:
            raise ValueError(f"Checksum mismatch: {name}")
        count += 1
    return count


def scalar_features(prefix_readouts):
    """Input is ONLY positions 0..t inclusive; std is population std (ddof=0)."""
    values = np.asarray(prefix_readouts, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (3, 3) or len(values) == 0:
        raise ValueError("Expected nonempty (positions, 3 layers, 3 metrics)")
    stats = np.stack((values[-1], values.mean(0), values.std(0),
                      values[-1] - values[0]), axis=-1)
    return stats.reshape(-1)


def iter_examples(root, t=32, parsed_only=False):
    """Yield available checkpoints only, one example per ID, batch-sized memory."""
    if t not in CHECKPOINTS:
        raise ValueError(f"t must be one of {CHECKPOINTS}")
    for batch in sorted(Path(root).glob("batch_[0-9][0-9][0-9][0-9]")):
        with (batch / "generations.jsonl").open(encoding="utf-8") as stream:
            records = {row["id"]: row for row in map(json.loads, stream)}
        with np.load(batch / "features.npz", allow_pickle=False) as archive:
            # NPZ indexing decompresses every access; materialize each array once per batch.
            data = {key: archive[key] for key in archive.files}
            readout_rows = {str(key): i for i, key in enumerate(data["readout_ids"])}
            for row in np.flatnonzero(data["checkpoints"] == t):
                problem_id = str(data["ids"][row])
                record = records[problem_id]
                if parsed_only and not record["label"]["analysis_eligible"]:
                    continue
                i = readout_rows[problem_id]
                start, end = map(int, data["readout_offsets"][i:i + 2])
                if end - start <= t:
                    raise ValueError(f"Missing prefix readouts for {problem_id}")
                yield {
                    "id": problem_id,
                    "y": int(not record["label"]["answer_correct"]),
                    "hidden": data["hidden"][row].copy(),
                    "trajectory_early": data["trajectory"][row, :14].copy(),
                    "readouts_prefix": data["readouts"][start:start + t + 1].copy(),
                }


def feature_matrix(root, t=32, kind="h14", parsed_only=False):
    ids, features, targets = [], [], []
    for example in iter_examples(root, t, parsed_only):
        h = example["hidden"]
        if kind == "h14":
            value = h[3]
        elif kind == "h14_delta":
            value = np.concatenate((h[3], h[3] - h[2]))
        elif kind == "multi":
            value = h[[0, 1, 3]].reshape(-1)
        elif kind == "scalar":
            value = scalar_features(example["readouts_prefix"])
        else:
            raise ValueError("kind must be h14, h14_delta, multi, or scalar")
        ids.append(example["id"])
        features.append(value)
        targets.append(example["y"])
    if not features:
        raise ValueError("No eligible examples")
    return np.asarray(ids), np.stack(features), np.asarray(targets, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Extracted gsm8k-train-all-v1 directory")
    parser.add_argument("--t", type=int, choices=CHECKPOINTS, default=32)
    parser.add_argument("--kind", choices=("h14", "h14_delta", "multi", "scalar"), default="h14")
    parser.add_argument("--parsed-only", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional derived matrix .npz (not a trained model)")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps({"verified_files": verify_checksums(args.root)}))
    ids, x, y = feature_matrix(args.root, args.t, args.kind, args.parsed_only)
    print(json.dumps({"t": args.t, "kind": args.kind, "shape": list(x.shape),
                      "correct": int((y == 0).sum()), "incorrect": int(y.sum()),
                      "parsed_only": args.parsed_only}))
    if args.output:
        np.savez_compressed(args.output, ids=ids, X=x, y=y)


if __name__ == "__main__":
    main()
