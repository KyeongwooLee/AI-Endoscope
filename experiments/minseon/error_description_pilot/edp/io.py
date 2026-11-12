"""Shared I/O helpers: hashing, LF-safe JSONL, atomic writes, fingerprints."""
import datetime as _dt
import hashlib
import json
import math
import os
import unicodedata
from pathlib import Path


class ProtocolError(RuntimeError):
    """Raised when inputs, fingerprints or protocol invariants do not match."""


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_text(text):
    """SHA256 of the exact UTF-8 encoding of a string, no normalization."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def object_sha256(value):
    return sha256_text(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False))


def normalize_question(text):
    """NFC Unicode normalization plus whitespace collapse (used for grouping only)."""
    return ' '.join(unicodedata.normalize('NFC', text).split())


def question_sha256(text):
    return sha256_text(normalize_question(text))


def read_jsonl(path):
    """Read JSONL by LF-delimited records only.

    Iterates over bytes split on b'\\n' so that U+2028/U+2029 or other
    Unicode line separators inside JSON strings never split a record
    (``str.splitlines`` would). Blank records are rejected rather than skipped
    silently, except a single trailing newline.
    """
    data = Path(path).read_bytes()
    parts = data.split(b'\n')
    if parts and parts[-1] == b'':
        parts.pop()
    rows = []
    for number, raw in enumerate(parts, 1):
        if raw.endswith(b'\r'):
            raw = raw[:-1]
        if not raw.strip():
            raise ProtocolError(f'{path}: blank JSONL record at line {number}')
        rows.append(json.loads(raw.decode('utf-8')))
    return rows


def dumps_line(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=False)


def _atomic_write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.tmp{os.getpid()}')
    with temp.open('wb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def write_text(path, text):
    _atomic_write_bytes(path, text.encode('utf-8'))


def write_json(path, value):
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_jsonl(path, rows):
    write_text(path, ''.join(dumps_line(r) + '\n' for r in rows))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(dumps_line(row) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def now_utc():
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def is_finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def guard_or_write_json(path, value, what):
    """Write ``value`` unless a different version already exists (refuse to mix)."""
    path = Path(path)
    if path.exists():
        existing = read_json(path)
        if existing != value:
            raise ProtocolError(f'{what} already exists with different content: {path}. '
                                'Use a new run_dir instead of mixing protocols.')
        return False
    write_json(path, value)
    return True
