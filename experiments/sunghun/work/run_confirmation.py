from pathlib import Path
import hashlib
import json
import subprocess
import time
import zipfile

root = Path('/tmp/gsm8k-confirmation-v1')
while not (root / 'READY.json').exists():
    time.sleep(5)
ready = json.loads((root / 'READY.json').read_text())
assert hashlib.sha256((root / 'confirmation-input.zip').read_bytes()).hexdigest() == ready['input_zip_sha256']
assert hashlib.sha256((root / 'confirm_entropy_features.py').read_bytes()).hexdigest() == ready['script_sha256']
with zipfile.ZipFile(root / 'confirmation-input.zip') as archive:
    assert all((root / name).resolve().is_relative_to(root) for name in archive.namelist())
    archive.extractall(root)
subprocess.run([str(root / 'env/bin/python'), '-u', 'confirm_entropy_features.py'], cwd=root, check=True)
