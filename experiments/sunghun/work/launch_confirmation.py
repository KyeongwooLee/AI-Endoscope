from pathlib import Path
import subprocess

root = Path('/tmp/gsm8k-confirmation-v1')
with (root / 'job.log').open('a') as stream:
    child = subprocess.Popen(['bash', 'bootstrap_confirmation.sh'], cwd=root,
        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
        close_fds=True, start_new_session=True)
print(child.pid)
