import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

root = Path('/tmp/gsm8k-train-all-v1')
ready = json.loads((root / 'READY.json').read_text())
package = root / 'train-all-input.zip'
assert hashlib.sha256(package.read_bytes()).hexdigest() == ready['package_sha256']
with zipfile.ZipFile(package) as archive:
    assert all((root / name).resolve().is_relative_to(root) for name in archive.namelist())
    archive.extractall(root)
environment = dict(__import__('os').environ)
environment.update(HF_HOME='/tmp/gsm8k-confirmation-v1/cache', HF_HUB_DISABLE_XET='1',
                   MPLCONFIGDIR=str(root / 'mpl'), CUBLAS_WORKSPACE_CONFIG=':4096:8')
with (root / 'job.log').open('a') as stream:
    child = subprocess.Popen(['/tmp/gsm8k-confirmation-v1/env/bin/python', '-u',
        'extract_train_all.py', '--root', str(root)], cwd=root, env=environment,
        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
        close_fds=True, start_new_session=True)
print(json.dumps({'pid': child.pid, 'detached': True}))
