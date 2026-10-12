"""Inventory and verified-backup acknowledgements for this experiment only."""
import argparse
import json
import os
from pathlib import Path
import re

ROOT = Path('/tmp/gsm8k-train-all-v1')
parser = argparse.ArgumentParser()
parser.add_argument('action', choices=('inventory', 'ack'))
parser.add_argument('--batch')
parser.add_argument('--sha256')
parser.add_argument('--bytes', type=int)
args = parser.parse_args()

if args.action == 'inventory':
    result = {'batches': []}
    for path in sorted((ROOT / 'outbox').glob('batch_*.json')):
        result['batches'].append({'name': path.stem, 'receipt': json.loads(path.read_text())})
    for name in ('manifest', 'progress', 'quality_gate', 'job_complete'):
        path = ROOT / (name + '.json')
        if path.exists():
            result[name] = json.loads(path.read_text())
    print(json.dumps(result))
else:
    if not re.fullmatch(r'batch_[0-9]{4}', args.batch or ''):
        raise ValueError('Invalid batch name')
    receipt = json.loads((ROOT / 'outbox' / (args.batch + '.json')).read_text())
    if receipt['archive_sha256'] != args.sha256 or receipt['archive_bytes'] != args.bytes:
        raise ValueError('Acknowledgement does not match committed archive')
    value = {'archive_sha256': args.sha256, 'archive_bytes': args.bytes}
    path = ROOT / 'acks' / (args.batch + '.json')
    path.parent.mkdir(exist_ok=True)
    partial = Path(str(path) + '.partial')
    partial.write_text(json.dumps(value) + '\n')
    os.replace(partial, path)
    target = (ROOT / 'outbox' / (args.batch + '.zip')).resolve()
    if target.parent != (ROOT / 'outbox').resolve() or not target.is_relative_to(ROOT.resolve()):
        raise ValueError('Unsafe cleanup target')
    target.unlink(missing_ok=True)
    print('BACKUP_ACKNOWLEDGED', args.batch)
