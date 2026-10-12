import subprocess
from pathlib import Path

root = Path('/tmp/gsm8k-feature-ablation-v1')
with (root / 'job.log').open('a') as stream:
    child = subprocess.Popen(['python', '-u', 'run_feature_ablation_remote.py'], cwd=root,
        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, close_fds=True, start_new_session=True)
print(child.pid)
