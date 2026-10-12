"""Independently verify a remote receipt snapshot and existing local batch backups."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument('--snapshot')
parser.add_argument('--host', required=True)
parser.add_argument('--port', required=True)
parser.add_argument('--local-run', required=True)
parser.add_argument('--ssh-bin', default=shutil.which('ssh') or 'ssh')
parser.add_argument('--scp-bin', default=shutil.which('scp') or 'scp')
args = parser.parse_args()
output = Path(args.local_run).resolve()
options = ['-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10']
host = args.host
remote = '/tmp/gsm8k-train-all-v1'
ssh, scp = args.ssh_bin, args.scp_bin


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def download(name, path):
    subprocess.run([scp, '-P', args.port, *options, host + ':' + remote + '/' + name, str(path)],
                   check=True, capture_output=True, timeout=300)


response = subprocess.run([ssh, '-p', args.port, *options, host,
    'python ' + remote + '/train_all_control.py inventory'], check=True, capture_output=True,
    text=True, timeout=90)
state = json.loads(response.stdout)
stamp = datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d-%H%M%S-KST')
backup = output / 'snapshots' / stamp
backup.mkdir(parents=True, exist_ok=False)
(backup / 'remote_inventory.json').write_text(json.dumps(state, indent=2))
summaries, batch_hashes, archive_bytes = [], {}, 0
for entry in state['batches']:
    name, receipt = entry['name'], entry['receipt']
    assert re.fullmatch(r'batch_[0-9]{4}', name)
    archive_path = output / 'incoming' / (name + '.zip')
    local_receipt = output / 'incoming' / (name + '.json')
    # The ongoing collector commits downloaded archives before its receipt.
    deadline = time.monotonic() + 90
    while not local_receipt.exists() and time.monotonic() < deadline:
        time.sleep(2)
    assert local_receipt.exists(), 'Collector has not received ' + name
    assert json.loads(local_receipt.read_text()) == receipt
    assert archive_path.stat().st_size == receipt['archive_bytes']
    assert sha(archive_path) == receipt['archive_sha256']
    with zipfile.ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == set(receipt['files'])
        assert len(archive.namelist()) == len(receipt['files'])
        for member, checksum in receipt['files'].items():
            assert (output / member).resolve().is_relative_to(output.resolve())
            assert hashlib.sha256(archive.read(member)).hexdigest() == checksum
            assert sha(output / member) == checksum
    batch_hashes[name] = receipt['archive_sha256']
    archive_bytes += receipt['archive_bytes']
    summaries.extend(receipt['summaries'])
ids = [r['id'] for r in summaries]
assert len(ids) == len(set(ids))
assert all(re.fullmatch(r'train-[0-9]{5}', identifier) for identifier in ids)
snapshot_value = None
if args.snapshot:
    assert re.fullmatch(r'snapshot-[0-9]{8}T[0-9]{6}Z', args.snapshot)
    receipt_path = backup / 'snapshot.receipt.json'
    download(args.snapshot + '.receipt.json', receipt_path)
    snapshot_value = json.loads(receipt_path.read_text())
    archive_path = backup / snapshot_value['archive']
    download(snapshot_value['archive'], archive_path)
    assert archive_path.stat().st_size == snapshot_value['archive_bytes']
    assert sha(archive_path) == snapshot_value['archive_sha256']
    with zipfile.ZipFile(archive_path) as archive:
        snapshot_manifest = json.loads(archive.read('snapshot_manifest.json'))
        for member, info in snapshot_manifest['members'].items():
            assert (backup / 'remote' / member).resolve().is_relative_to((backup / 'remote').resolve())
            data = archive.read(member)
            assert len(data) == info['bytes'] and hashlib.sha256(data).hexdigest() == info['sha256']
        assert set(archive.namelist()) == set(snapshot_manifest['members']) | {'snapshot_manifest.json'}
        archive.extractall(backup / 'remote')
new = [r for r in summaries if r['origin'] == 'new_train']
result = {'verified_at_kst': datetime.now(timezone(timedelta(hours=9))).isoformat(),
          'verified_batches': len(batch_hashes), 'verified_problems': len(summaries),
          'batch_archive_bytes': archive_bytes, 'incorrect_in_verified_pool': sum(r['incorrect'] for r in summaries),
          'new_generations_backed_up': len(new), 'new_incorrect_backed_up': sum(r['incorrect'] for r in new),
          'remote_progress_at_inventory': state.get('progress'), 'snapshot': snapshot_value,
          'all_archive_member_and_extracted_hashes_passed': True, 'batch_hashes': batch_hashes}
(backup / 'verification_receipt.json').write_text(json.dumps(result, indent=2))
(output / 'latest_backup_audit.json').write_text(json.dumps({
    **{k: v for k, v in result.items() if k != 'batch_hashes'},
    'receipt_path': str(backup / 'verification_receipt.json')}, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != 'batch_hashes'}, indent=2))
