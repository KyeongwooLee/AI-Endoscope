import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import urllib.request
from pathlib import Path


ASSET_HASHES = {
    'errors.jsonl': 'cc54b199164cdef0c42365a1b806f3dc1e99580762d3b6976dabfba3d2c1993b',
    'token_ids.jsonl': '10163b60dad0526c07123197227f42bae4e8089ea040018cb533183e326ed49e',
    'features_meta.json': 'b0d6ba830ff66432dd118f3c0550db4d7d7a281e08111b431b225a37fbe60464',
    'confidence.npy': '564008b2493f4c99840c6946444251a6b0fba22478d2be1413a20dba9ba6db1f',
    'DATA_SCHEMA.json': 'e1a62842dbc54a74a2691e3531dcc2b1af1cc211125645c44a7b1d8b207a9a13',
    'README.md': 'b239884f998c6724949e00bb16b090dc21434ffd2a5749a4bca2c98eca7f9316',
    'SHA256SUMS.txt': '2b77c01a9e718b67cb74eaa07a0052f884c422bec5daef625fc16be1d461c829',
    'ATTRIBUTION.md': '426e5d902a316d771d6620e079a3ee7d044c55611ade27e6cfae8911101ef254',
    'LICENSE-GSM8K.txt': '893951b3bf94db8df1b13e05da5cdeb499400960e4d44a3962a8b33ed0b4f28e',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_rows(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def atomic_npz(path, **arrays):
    import numpy as np
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temp, path)


@contextlib.contextmanager
def lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / '.lock'
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(fd, 'w') as handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        path.unlink()


def paths(cfg):
    return Path(cfg['data_dir']), Path(cfg['output_dir'])


def download(url, target, expected):
    target = Path(target)
    if target.exists():
        if digest(target) != expected:
            raise ValueError(f'Existing file checksum mismatch; preserve and investigate: {target}')
        return
    temp = target.with_name(target.name + '.download')
    request = urllib.request.Request(url, headers={'User-Agent': 'YAICON-SAE-clustering'})
    with urllib.request.urlopen(request, timeout=120) as response, temp.open('wb') as handle:
        while block := response.read(1024 * 1024):
            handle.write(block)
    if digest(temp) != expected:
        raise ValueError(f'Download checksum mismatch: {temp}')
    os.replace(temp, target)


def prepared(cfg):
    _, out = paths(cfg)
    manifest = read_json(out / 'manifest.json')
    if manifest['config'] != cfg or manifest['prepared_sha256'] != digest(out / 'prepared.jsonl'):
        raise ValueError('Prepared inputs or config changed; use a new output directory')
    return read_rows(out / 'prepared.jsonl'), manifest


def runtime_signature(cfg, manifest):
    versions = {}
    for name in ['torch', 'transformers', 'numpy', 'scikit-learn', 'safetensors', 'huggingface-hub']:
        versions[name] = importlib.metadata.version(name)
    code = {p.name: digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    return {'config': cfg, 'prepared_sha256': manifest['prepared_sha256'],
            'python': platform.python_version(), 'packages': versions, 'code': code}
