import hashlib
import json
import os
from pathlib import Path
import tempfile


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_digest(names=None):
    root = Path(__file__).parent
    paths = sorted(root.glob("*.py")) if names is None else [root / name for name in sorted(names)]
    return digest({p.name: file_digest(p) for p in paths})


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp = Path(name)
    try:
        writer(temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_json(path, value):
    atomic_write(path, lambda p: p.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8"))


def write_text(path, value):
    atomic_write(path, lambda p: p.write_text(value, encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    write_text(path, "".join(json.dumps(row, ensure_ascii=False, allow_nan=False)
                            + "\n" for row in rows))


def write_parquet(path, frame):
    atomic_write(path, lambda p: frame.to_parquet(p, index=False))


def commit_files(directory, filenames, fingerprint):
    directory = Path(directory)
    write_json(directory / "complete.json", {
        "fingerprint": fingerprint,
        "files": {name: file_digest(directory / name) for name in filenames},
    })


def complete(directory, fingerprint, verify=False):
    directory = Path(directory)
    marker = directory / "complete.json"
    if not marker.exists():
        return False
    record = read_json(marker)
    if record["fingerprint"] != fingerprint:
        raise ValueError(f"Incompatible artifact: {directory}. Use a new run directory.")
    for name, checksum in record["files"].items():
        path = directory / name
        if not path.is_file() or (verify and file_digest(path) != checksum):
            raise ValueError(f"Missing/corrupt completed artifact: {path}")
    return True
