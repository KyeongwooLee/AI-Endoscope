"""Content-addressed stage receipts and atomic, resumable problem shards."""

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory


def json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json_bytes(value) + b"\n")
    temporary.replace(path)


def index(rows, field="id"):
    result = {}
    for row in rows:
        key = row[field]
        if key in result:
            raise ValueError(f"Duplicate {field}: {key}")
        result[key] = row
    return result


def environment(device):
    packages = ("torch", "transformers", "huggingface-hub", "math-verify", "antlr4-python3-runtime",
                "numpy", "scikit-learn", "pyarrow", "sympy", "tokenizers", "safetensors",
                "latex2sympy2_extended", "scipy")
    import torch
    cuda = ({"name": torch.cuda.get_device_name(device),
             "capability": list(torch.cuda.get_device_capability(device)), "runtime": torch.version.cuda,
             "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version",
                                                "--format=csv,noheader"], text=True).splitlines()[0]}
            if str(device).startswith("cuda") else None)
    return {"python": platform.python_version(), "os": platform.platform(), "gpu": cuda,
            "packages": {name: importlib.metadata.version(name) for name in packages}}


def source():
    package = Path(__file__).parent
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=package, text=True).strip()
    return {"git_head": head, "files": {p.name: sha256(p) for p in sorted(package.glob("*.py"))},
            "minsu": "d95116f0d8a6b64a7f3eb08457af11144d6f4a7c",
            "sunghun": "99b35512ba043b0ec86d809ec700fcb4ba969889",
            "research_base": None}


class Stage:
    def __init__(self, root, name):
        self.path = root / name
        self.receipt = self.path / "manifest.json"
        self.units = {}

    def shard(self, key):
        return self.path / digest(key)

    def read_unit(self, key):
        if key not in self.units:
            raise ValueError(f"Missing stage unit {key} in {self.path}")
        return self.shard(key)

    def _verify(self, key):
        directory = self.shard(key)
        receipt = read(directory / "receipt.json")
        observed = {p.name: sha256(p) for p in directory.iterdir()
                    if p.is_file() and p.name != "receipt.json"}
        if receipt != {"key": key, "files": observed}:
            raise ValueError(f"Invalid shard checksum: {directory}")
        return digest(receipt)

    def load(self):
        receipt = read(self.receipt)
        self.units = receipt["outputs"]
        for key, expected in self.units.items():
            if self._verify(key) != expected:
                raise ValueError(f"Stage output changed: {self.path}/{key}")
        return receipt

    def begin(self, identity, resume):
        marker = self.path / "inputs.json"
        if self.path.exists():
            if not resume or not marker.exists() or read(marker) != identity:
                raise ValueError(f"Cannot resume {self.path}: use --resume with identical inputs and config, or a new run directory")
            if self.receipt.exists():
                if self.load()["inputs"] != identity:
                    raise ValueError(f"Stage manifest inputs changed: {self.path}")
                return False
        else:
            self.path.mkdir()
            write(marker, identity)
        for partial in self.path.glob(".partial-*"):
            shutil.rmtree(partial)
        self.identity = identity
        return True

    def unit(self, key, build):
        if key in self.units:
            raise ValueError(f"Duplicate output unit: {key}")
        directory = self.shard(key)
        if not directory.exists():
            with TemporaryDirectory(prefix=".partial-", dir=self.path) as temp:
                scratch = Path(temp)
                build(scratch)
                write(scratch / "receipt.json", {"key": key, "files": {
                    p.name: sha256(p) for p in scratch.iterdir() if p.is_file()}})
                scratch.rename(directory)
        self.units[key] = self._verify(key)
        if len(self.units) % 20 == 0:
            print(f"{self.path.name}: {len(self.units)} units", flush=True)
        return directory

    def finish(self):
        write(self.receipt, {"inputs": self.identity, "outputs": self.units})
