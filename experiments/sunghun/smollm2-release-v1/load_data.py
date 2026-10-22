"""CPU-only compact/full loaders. NumPy pickle is never enabled."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np

CHECKPOINTS = (0, 32, 64, 128)
RAW_LAYERS = (6, 9, 11, 12)
READOUT_LAYERS = (6, 9, 12)
READOUT_METRICS = ("entropy", "margin", "top1_probability")
STATS = ("current", "mean", "std", "change_from_t0")
SCALAR_NAMES = tuple(f"l{layer}.{metric}.{stat}" for layer in READOUT_LAYERS
                     for metric in READOUT_METRICS for stat in STATS)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
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
            raise ValueError("Unsafe checksum path")
        if sha256(path) != expected:
            raise ValueError("Checksum mismatch: " + name)
        count += 1
    return count


def scalar_features(prefix):
    values = np.asarray(prefix, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (3, 3) or not len(values):
        raise ValueError("Expected nonempty (positions,3,3)")
    return np.stack([values[-1], values.mean(0), values.std(0), values[-1] - values[0]], -1).reshape(-1)


def iter_compact(root, t=32, parsed_only=False):
    if t not in CHECKPOINTS:
        raise ValueError("Unavailable checkpoint")
    for batch in sorted(Path(root).glob("batch_[0-9][0-9][0-9][0-9]")):
        records = {r["id"]: r for r in map(json.loads, (batch / "generations.jsonl").read_text(encoding="utf-8").splitlines())}
        with np.load(batch / "features.npz", allow_pickle=False) as z:
            data = {key: z[key] for key in z.files}
        lookup = {str(v): i for i, v in enumerate(data["readout_ids"])}
        for index in np.flatnonzero(data["checkpoints"] == t):
            identifier = str(data["ids"][index])
            record = records[identifier]
            if parsed_only and not record["label"]["analysis_eligible"]:
                continue
            r = lookup[identifier]
            start, end = map(int, data["readout_offsets"][r:r + 2])
            if end - start <= t:
                raise ValueError("Missing prefix for " + identifier)
            yield dict(id=identifier, y=int(not record["label"]["answer_correct"]),
                hidden=data["hidden"][index].copy(), trajectory_early=data["trajectory"][index, :12].copy(),
                readouts_prefix=data["readouts"][start:start + t + 1].copy())


def feature_matrix(root, t=32, kind="middle", parsed_only=False):
    ids, x, y = [], [], []
    for example in iter_compact(root, t, parsed_only):
        h = example["hidden"]
        if kind == "middle":
            value = h[3]
        elif kind == "middle_delta":
            value = np.concatenate([h[3], h[3] - h[2]])
        elif kind == "multi":
            value = h[[0, 1, 3]].reshape(-1)
        elif kind == "scalar":
            value = scalar_features(example["readouts_prefix"])
        else:
            raise ValueError("Unknown feature kind")
        ids.append(example["id"])
        x.append(value)
        y.append(example["y"])
    if not x:
        raise ValueError("No examples at the requested checkpoint")
    return np.asarray(ids), np.stack(x), np.asarray(y, np.int64)


def iter_full(root, dataset=None):
    """Each full part contains original batch ZIPs; only one example is materialized."""
    for path in sorted((Path(root) / "batches").glob("batch_*.zip")):
        with zipfile.ZipFile(path) as z:
            for name in sorted(n for n in z.namelist() if n.startswith("features/") and n.endswith(".npz")):
                with np.load(io.BytesIO(z.read(name)), allow_pickle=False) as archive:
                    record = json.loads(str(archive["record"]))
                    if dataset is not None and record["dataset"] != dataset:
                        continue
                    yield record, {k: archive[k] for k in archive.files if k != "record"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--t", type=int, choices=CHECKPOINTS, default=32)
    p.add_argument("--kind", choices=["middle", "middle_delta", "multi", "scalar"], default="middle")
    p.add_argument("--parsed-only", action="store_true")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.verify:
        print(json.dumps(dict(verified_files=verify_checksums(args.root))))
    ids, x, y = feature_matrix(args.root, args.t, args.kind, args.parsed_only)
    print(json.dumps(dict(shape=list(x.shape), correct=int((y == 0).sum()), wrong=int(y.sum()))))
    if args.output:
        np.savez_compressed(args.output, ids=ids, X=x, y=y)


if __name__ == "__main__":
    main()
