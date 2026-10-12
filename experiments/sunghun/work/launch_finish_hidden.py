import os
from pathlib import Path
import subprocess

root = Path('/tmp/gsm8k-hidden-study')
with (root / 'analysis.log').open('a') as stream:
    process = subprocess.Popen(['python', '-u', 'finish_hidden_remote.py'], cwd=root, env=os.environ.copy(),
        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
        close_fds=True, start_new_session=True)
print(process.pid)
