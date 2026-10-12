#!/bin/bash
set -euo pipefail
cd /tmp/gsm8k-hidden-study
export HF_HOME=/tmp/gsm8k-hidden-study/cache
export MPLCONFIGDIR=/tmp/gsm8k-hidden-study/mpl
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m venv env
env/bin/python -m pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
env/bin/python -m pip install --no-cache-dir transformers==4.57.6 huggingface-hub==0.36.0 safetensors==0.7.0 numpy==2.2.6 pandas==2.3.3 pyarrow==22.0.0 matplotlib==3.10.8 scikit-learn==1.7.2
env/bin/python -c "import zipfile; zipfile.ZipFile('hidden_input.zip').extractall('input')"
env/bin/python -u extract_hidden_probes.py --input input/examples.jsonl --output extracted
env/bin/python -u train_hidden_probes.py --data extracted --splits input/splits.json --output results
env/bin/python -u analyze_hidden_trajectory.py --data extracted --output results
env/bin/python -c "from pathlib import Path; import hashlib,json; r=Path('.'); ps=[p for d in ('extracted','results','input') for p in (r/d).rglob('*') if p.is_file()]; ps += [p for p in r.glob('*.py')]; m={str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in ps}; Path('backup_manifest.json').write_text(json.dumps(m,indent=2))"
env/bin/python -c "import zipfile,json; from pathlib import Path; m=json.loads(Path('backup_manifest.json').read_text()); z=zipfile.ZipFile('hidden-study-backup.zip','w',zipfile.ZIP_DEFLATED); [z.write(p,p) for p in [*m,'backup_manifest.json','job.log'] if Path(p).exists()]; z.close()"
echo ALL_STAGES_COMPLETE
