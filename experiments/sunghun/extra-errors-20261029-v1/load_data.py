"""Read original nested checkpoints; LF-only JSONL and NumPy without pickle."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile


def jsonl(data):
    for line in data.decode('utf-8').split('\n'):
        if line.strip():
            yield json.loads(line)


def outbox(root):
    path = Path(root) / 'backup/outbox'
    if not path.is_dir():
        raise FileNotFoundError('Extract all full-part ZIPs into the same directory first: ' + str(path))
    return path


def iter_generations(root):
    for p in sorted(outbox(root).glob('generations_*.zip')):
        with zipfile.ZipFile(p) as z:
            yield from jsonl(z.read('records.jsonl'))


def iter_features(root):
    import numpy as np
    for p in sorted(outbox(root).glob('features_*.zip')):
        with zipfile.ZipFile(p) as z:
            for record in jsonl(z.read('records.jsonl')):
                with np.load(io.BytesIO(z.read('features/' + record['id'] + '.npz')), allow_pickle=False) as a:
                    yield record, {name: a[name] for name in a.files}


def verify(root):
    counts = {'generations': 0, 'features': 0}
    for p in sorted(outbox(root).glob('*.json')):
        receipt = json.loads(p.read_text(encoding='utf-8'))
        archive = p.with_suffix('.zip')
        h = hashlib.sha256()
        with archive.open('rb') as f:
            for chunk in iter(lambda: f.read(4 * 1024**2), b''):
                h.update(chunk)
        if archive.stat().st_size != receipt['bytes'] or h.hexdigest() != receipt['archive_sha256']:
            raise ValueError('Original ZIP SHA256/size mismatch: ' + str(archive))
        with zipfile.ZipFile(archive) as z:
            if set(z.namelist()) != set(receipt['members']):
                raise ValueError('Member set mismatch: ' + str(archive))
            for name, expected in receipt['members'].items():
                if hashlib.sha256(z.read(name)).hexdigest() != expected:
                    raise ValueError('Member hash mismatch: ' + name)
            records = list(jsonl(z.read('records.jsonl')))
            if [r['id'] for r in records] != receipt['ids']:
                raise ValueError('Receipt ID mismatch: ' + str(archive))
            counts['generations' if p.name.startswith('generations_') else 'features'] += len(records)
    if counts != {'generations': 18656, 'features': 4000}:
        raise ValueError('Incomplete full dataset: ' + str(counts))
    return counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(args.root)))
    else:
        record, arrays = next(iter_features(args.root))
        print(record['id'], {k: list(v.shape) for k, v in arrays.items()})
