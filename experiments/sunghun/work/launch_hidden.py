import os
from pathlib import Path
import subprocess

root = Path('/tmp/gsm8k-hidden-study')
env = dict(os.environ, HF_HOME=str(root / 'cache'), MPLCONFIGDIR=str(root / 'mpl'),
           CUBLAS_WORKSPACE_CONFIG=':4096:8', HF_HUB_DISABLE_XET='1')
with (root / 'extract.log').open('a') as stream:
    process = subprocess.Popen([str(root / 'env/bin/python'), '-u', 'extract_hidden_probes.py',
        '--input', 'input/examples.jsonl', '--output', 'extracted'], cwd=root, env=env,
        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
        close_fds=True, start_new_session=True)
print(process.pid)
