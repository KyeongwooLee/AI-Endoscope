"""Receive compact batches, verify every member, then acknowledge remote cleanup."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import zipfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--host', required=True)
parser.add_argument('--port', required=True)
parser.add_argument('--local-run', required=True)
parser.add_argument('--ssh-bin', default=shutil.which('ssh') or 'ssh')
parser.add_argument('--scp-bin', default=shutil.which('scp') or 'scp')
parser.add_argument('--hours', type=float, default=72)
args = parser.parse_args()
output = Path(args.local_run).resolve()
incoming = output / 'incoming'
incoming.mkdir(parents=True, exist_ok=True)
ssh, scp = args.ssh_bin, args.scp_bin
host = args.host
remote = '/tmp/gsm8k-train-all-v1'
options = ['-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10']
verified = set()


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    temp = Path(str(path) + '.partial')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def remote_command(command):
    return subprocess.run([ssh, '-p', args.port, *options, host, command],
                          check=True, capture_output=True, text=True, timeout=90).stdout


def copy_remote(name, destination):
    subprocess.run([scp, '-P', args.port, *options, host + ':' + remote + '/' + name, str(destination)],
                   check=True, capture_output=True, timeout=300)


def verify_batch(archive, receipt, extract):
    assert archive.stat().st_size == receipt['archive_bytes'] and sha(archive) == receipt['archive_sha256']
    with zipfile.ZipFile(archive) as packed:
        assert len(packed.namelist()) == len(receipt['files'])
        assert set(packed.namelist()) == set(receipt['files'])
        for name, checksum in receipt['files'].items():
            assert (output / name).resolve().is_relative_to(output.resolve())
            assert hashlib.sha256(packed.read(name)).hexdigest() == checksum
        if extract:
            packed.extractall(output)


started = time.monotonic()
while time.monotonic() - started < args.hours * 3600:
    try:
        state = json.loads(remote_command('python ' + remote + '/train_all_control.py inventory'))
        for entry in state['batches']:
            name, receipt = entry['name'], entry['receipt']
            assert re.fullmatch(r'batch_[0-9]{4}', name)
            local_receipt = incoming / (name + '.json')
            local_ack = incoming / (name + '.ack.json')
            archive = incoming / (name + '.zip')
            if name not in verified:
                if local_receipt.exists():
                    assert json.loads(local_receipt.read_text()) == receipt
                    verify_batch(archive, receipt, extract=True)
                else:
                    partial = Path(str(archive) + '.partial')
                    copy_remote('outbox/' + name + '.zip', partial)
                    verify_batch(partial, receipt, extract=True)
                    os.replace(partial, archive)
                    write_json(local_receipt, receipt)
                    print('LOCAL_BATCH_VERIFIED', name, len(receipt['ids']), flush=True)
                verified.add(name)
            if not local_ack.exists():
                response = remote_command('python ' + remote + '/train_all_control.py ack --batch ' + name
                    + ' --sha256 ' + receipt['archive_sha256'] + ' --bytes ' + str(receipt['archive_bytes']))
                assert 'BACKUP_ACKNOWLEDGED ' + name in response
                write_json(local_ack, {'archive_sha256': receipt['archive_sha256'], 'remote_archive_removed': True})
        for name in ('manifest', 'progress', 'quality_gate', 'job_complete'):
            if name in state:
                write_json(output / (name + '.json'), state[name])
        write_json(output / 'collector_status.json', {'state': 'receiving', 'verified_batches': len(verified),
                   'verified_problems': sum(len(e['receipt']['ids']) for e in state['batches'] if e['name'] in verified),
                   'time_unix': time.time()})
        if 'job_complete' in state:
            completion = state['job_complete']
            assert len(verified) == completion['batches']
            assert sum(len(e['receipt']['ids']) for e in state['batches']) == 7473
            archive = output / 'metadata-backup.zip'
            partial = Path(str(archive) + '.partial')
            copy_remote('metadata-backup.zip', partial)
            assert sha(partial) == completion['archive_sha256']
            assert partial.stat().st_size == completion['archive_bytes']
            with zipfile.ZipFile(partial) as packed:
                manifest = json.loads(packed.read('backup_manifest.json'))
                for name, info in manifest.items():
                    assert (output / name).resolve().is_relative_to(output.resolve())
                    content = packed.read(name)
                    assert len(content) == info['bytes'] and hashlib.sha256(content).hexdigest() == info['sha256']
                assert all((output / name).resolve().is_relative_to(output.resolve()) for name in packed.namelist())
                packed.extractall(output)
            os.replace(partial, archive)
            write_json(output / 'local_backup_receipt.json', {'metadata_sha256': completion['archive_sha256'],
                       'verified_batches': len(verified), 'verified_problems': 7473,
                       'batch_hashes': {e['name']: e['receipt']['archive_sha256'] for e in state['batches']}})
            write_json(output / 'collector_status.json', {'state': 'complete', 'verified_problems': 7473})
            print('ALL_TRAIN_BACKUPS_VERIFIED', flush=True)
            break
    except Exception as error:
        write_json(output / 'collector_status.json', {'state': 'retrying', 'error': repr(error), 'time_unix': time.time()})
        print('COLLECTOR_RETRY', repr(error), flush=True)
    time.sleep(30)
