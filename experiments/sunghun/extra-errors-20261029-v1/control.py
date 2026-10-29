import json
from pathlib import Path
import sys

root=Path(__file__).resolve().parent
if sys.argv[1:]!=['inventory']: raise SystemExit('inventory only')
metadata={}
for name in ['progress.json','summary.json','selected_ids.json','failure.json','environment.json',
             'run_manifest.json','model_files.json','input/input_summary.json','feature_schema.json']:
    p=root/name
    if p.is_file(): metadata[name]=json.loads(p.read_text(encoding='utf-8'))
receipts=[]
for path in sorted((root/'outbox').glob('*.json')):
    record=json.loads(path.read_text(encoding='utf-8'))
    assert record['name']==path.stem
    if path.with_suffix('.zip').is_file(): receipts.append(record)
print(json.dumps({'metadata':metadata,'receipts':receipts}))
