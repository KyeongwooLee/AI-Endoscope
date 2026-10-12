"""Range-download the pinned model and verify its LFS SHA before cache installation."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import time

from huggingface_hub import get_hf_file_metadata, hf_hub_url
import requests

ROOT = Path('/tmp/gsm8k-train-all-v1')
config = json.loads((ROOT / 'input/config.json').read_text())
revision = config['model']['revision']
metadata = get_hf_file_metadata(hf_hub_url(config['model']['id'], 'model.safetensors', revision=revision))
assert metadata.commit_hash == revision and re.fullmatch('[0-9a-f]{64}', metadata.etag)
hub = Path(os.environ['HF_HOME']) / 'hub' / 'models--Qwen--Qwen2.5-1.5B-Instruct'
blob = hub / 'blobs' / metadata.etag


def checksum(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            value.update(chunk)
    return value.hexdigest()


target = ROOT / 'model.parallel.partial'
size = metadata.size
if not blob.exists() or blob.stat().st_size != size or checksum(blob) != metadata.etag:
    fd = os.open(target, os.O_CREAT | os.O_RDWR, 0o644)
    os.ftruncate(fd, size)
    parts = [(start, min(size - 1, start + 8 * 2**20 - 1)) for start in range(0, size, 8 * 2**20)]
    started = time.monotonic()

    def download(part):
        start, end = part
        for attempt in range(5):
            try:
                with requests.get(metadata.location, headers={'Range': f'bytes={start}-{end}'},
                                  timeout=(20, 60)) as response:
                    response.raise_for_status()
                    assert response.status_code == 206
                    assert response.headers.get('Content-Range') == f'bytes {start}-{end}/{size}'
                    data = response.content
                    assert len(data) == end - start + 1
                    offset = 0
                    while offset < len(data):
                        written = os.pwrite(fd, data[offset:], start + offset)
                        assert written > 0
                        offset += written
                return len(data)
            except Exception as error:
                if attempt == 4:
                    raise RuntimeError(f'Model range {start}:{end} failed: {type(error).__name__}') from error
                time.sleep(attempt + 1)

    try:
        completed = 0
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(download, part) for part in parts]
            for index, future in enumerate(as_completed(futures), 1):
                completed += future.result()
                if index % 16 == 0 or index == len(parts):
                    print(json.dumps({'model_download_mb': round(completed / 1e6),
                                      'total_mb': round(size / 1e6),
                                      'mb_per_second': round(completed / 1e6 / (time.monotonic() - started), 2)}), flush=True)
        os.fsync(fd)
    finally:
        os.close(fd)
    assert checksum(target) == metadata.etag, 'Full model SHA-256 mismatch'
    blob.parent.mkdir(parents=True, exist_ok=True)
    os.replace(target, blob)
snapshot = hub / 'snapshots' / revision / 'model.safetensors'
snapshot.parent.mkdir(parents=True, exist_ok=True)
if not snapshot.exists():
    snapshot.symlink_to(Path('../../blobs') / metadata.etag)
result = {'verified': True, 'sha256': metadata.etag, 'bytes': size, 'revision': revision}
(ROOT / 'model_download_verified.json').write_text(json.dumps(result, indent=2))
print('PINNED_MODEL_SHA_VERIFIED', json.dumps(result), flush=True)
