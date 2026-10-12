import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

root = Path('/tmp/gsm8k-feature-ablation-v1')
env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8', MPLCONFIGDIR=str(root / 'mpl'))
with zipfile.ZipFile(root / 'input.zip') as archive:
    assert all((root / 'input' / p).resolve().is_relative_to((root / 'input').resolve()) for p in archive.namelist())
    archive.extractall(root / 'input')
subprocess.run(['/tmp/gsm8k-hidden-study/env/bin/python', '-u', 'train_feature_ablation.py',
    '--input', 'input', '--output', 'results'], cwd=root, env=env, check=True)
files = [p for d in ('input', 'results') for p in (root / d).rglob('*') if p.is_file()]
files += list(root.glob('*.py'))
manifest = {str(p.relative_to(root)): {'bytes': p.stat().st_size,
    'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
(root / 'backup_manifest.json').write_text(json.dumps(manifest, indent=2))
with zipfile.ZipFile(root / 'backup.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for p in files + [root / 'backup_manifest.json']:
        archive.write(p, str(p.relative_to(root)))
archive_path = root / 'backup.zip'
(root / 'job_complete.json').write_text(json.dumps({'archive_bytes': archive_path.stat().st_size,
    'archive_sha256': hashlib.sha256(archive_path.read_bytes()).hexdigest()}, indent=2))
print('ALL_STAGES_COMPLETE', flush=True)
