import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

root = Path('/tmp/gsm8k-hidden-study')
pid = 1369
env = dict(os.environ, HF_HOME=str(root / 'cache'), MPLCONFIGDIR=str(root / 'mpl'),
           CUBLAS_WORKSPACE_CONFIG=':4096:8')
while not (root / 'extracted/complete.json').exists():
    proc = Path(f'/proc/{pid}/stat')
    if not proc.exists() or proc.read_text().split()[2] == 'Z':
        raise RuntimeError('Extraction stopped without completion; inspect extract.log')
    time.sleep(15)
for script in ('train_hidden_probes.py', 'analyze_hidden_trajectory.py'):
    command = [str(root / 'env/bin/python'), '-u', script, '--data', 'extracted', '--output', 'results']
    if script.startswith('train'):
        command.extend(['--splits', 'input/splits.json'])
    subprocess.run(command, cwd=root, env=env, check=True)
files = [p for directory in ('extracted', 'results', 'input') for p in (root / directory).rglob('*') if p.is_file()]
files += list(root.glob('*.py')) + [root / 'extract.log']
manifest = {str(p.relative_to(root)): {'bytes': p.stat().st_size,
    'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
(root / 'backup_manifest.json').write_text(json.dumps(manifest, indent=2))
with zipfile.ZipFile(root / 'hidden-study-backup.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in [*files, root / 'backup_manifest.json']:
        archive.write(path, str(path.relative_to(root)))
archive_path = root / 'hidden-study-backup.zip'
(root / 'job_complete.json').write_text(json.dumps({'archive_bytes': archive_path.stat().st_size,
    'archive_sha256': hashlib.sha256(archive_path.read_bytes()).hexdigest()}, indent=2))
print('ALL_STAGES_COMPLETE', flush=True)
