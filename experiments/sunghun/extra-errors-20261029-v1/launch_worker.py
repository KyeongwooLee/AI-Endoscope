import json
import os
from pathlib import Path
import subprocess
import sys
import time

from protocol import sha,write_json

root=Path(__file__).resolve().parent
python=root/'venv/bin/python'
assert python.is_file()
assert (root/'model_files.json').is_file(), 'Pinned model download is not complete'
assert (root/'baseline_dedup.json').is_file(), 'Historical response deduplication input is not ready'
old_pid=root/'worker_pid.json'
if old_pid.exists():
    previous=json.loads(old_pid.read_text())['pid']
    try: os.kill(previous,0)
    except ProcessLookupError: pass
    else:
        stat=Path(f'/proc/{previous}/stat')
        zombie=stat.exists() and stat.read_text().rsplit(')',1)[-1].split()[0]=='Z'
        if not zombie: raise SystemExit('Worker PID is still alive; refusing duplicate producer')
baseline=json.loads((root/'baseline_dedup.json').read_text())
summary=json.loads((root/'input/input_summary.json').read_text())
summary.update(existing_response_pairs=baseline['total_known_pairs'],historical_greedy_rows=baseline['old_greedy_rows'],
               historical_greedy_zip_sha256=baseline['baseline_zip_sha256'])
write_json(root/'input/input_summary.json',summary)
files=['worker.py','protocol.py','control.py','input/config.json','input/questions.json','input/existing_pairs.json',
       'input/input_summary.json','baseline_dedup.json','model_files.json']
manifest={'files':{name:sha(root/name) for name in files},'created_unix':time.time(),
          'remote_root':str(root),'source_commit':subprocess.check_output(['git','-C',str(root.parent),'rev-parse','HEAD'],text=True).strip()}
if not (root/'run_manifest.json').exists(): write_json(root/'run_manifest.json',manifest)
else:
    old=json.loads((root/'run_manifest.json').read_text())
    assert old['files']==manifest['files'],'Frozen source/input changed'
env=os.environ.copy()
env['HF_HUB_DISABLE_XET']='1'
env['TOKENIZERS_PARALLELISM']='false'
log=(root/'worker.log').open('ab',buffering=0)
args=[str(python),'-u',str(root/'worker.py')]+(['--smoke'] if '--smoke' in sys.argv else [])
p=subprocess.Popen(args,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
write_json(old_pid,{'pid':p.pid,'smoke':'--smoke' in sys.argv,'started_unix':time.time()})
print(json.dumps({'pid':p.pid,'smoke':'--smoke' in sys.argv,'root':str(root)}))
