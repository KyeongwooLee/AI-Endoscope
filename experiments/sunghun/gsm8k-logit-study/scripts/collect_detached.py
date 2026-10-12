"""Reconnectable local collector; server inference does not depend on this process."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filelock import FileLock
from gsm8k_lens.storage import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', required=True)
    parser.add_argument('--remote-project', required=True)
    parser.add_argument('--remote-run', required=True)
    parser.add_argument('--local-run', required=True)
    parser.add_argument('--ssh-bin', default='ssh')
    parser.add_argument('--scp-bin', default='scp')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    run = Path(args.local_run).resolve()
    command = [sys.executable, '-u', str(project / 'scripts/sync_from_vessl.py'),
               '--queue-only', '--phase', 'test', '--batches', '20']
    for name in ('host', 'port', 'remote_project', 'remote_run', 'local_run', 'ssh_bin', 'scp_bin'):
        command += ['--' + name.replace('_', '-'), getattr(args, name)]
    status = {'state': 'running', 'phase': 'test', 'mode': 'detached_server',
              'remote_run': args.remote_run, 'local_run': str(run)}
    def save(**values):
        status.update(values, updated_unix=time.time())
        write_json(run / 'job_status.json', status)
    with FileLock(str(run / '.job.lock'), timeout=0):
        try:
            while True:
                save(step='collect_queued_results')
                result = subprocess.run(command, cwd=project, capture_output=True, text=True)
                print(result.stdout, end='', flush=True)
                print(result.stderr, end='', file=sys.stderr, flush=True)
                if result.returncode:
                    transient = any(word in result.stderr.lower() for word in
                                    ('connection reset', 'connection timed out', 'connection refused',
                                     'connection closed', 'no route to host', 'could not resolve hostname',
                                     'broken pipe', 'connection aborted', 'connection to '))
                    if not transient:
                        raise RuntimeError('Collector failed; remote producer is independent. Inspect collector stderr.')
                    save(state='reconnecting', step='waiting_for_ssh_server_keeps_running')
                    time.sleep(60)
                    continue
                save(state='running')
                if 'Phase transferred.' in result.stdout:
                    save(step='analyze')
                    subprocess.run([sys.executable, '-m', 'gsm8k_lens', 'analyze',
                                    '--run', str(run), '--phase', 'test'], cwd=project, check=True)
                    save(state='complete', step='complete')
                    return
                time.sleep(30)
        except Exception as error:
            save(state='failed', error=str(error))
            raise


if __name__ == '__main__':
    main()
