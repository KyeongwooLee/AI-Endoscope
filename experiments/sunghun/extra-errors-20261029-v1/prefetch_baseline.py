import hashlib
import json
import os
from pathlib import Path
import unicodedata
import zipfile
import requests

root=Path(__file__).resolve().parent
path=root/'baseline_greedy.zip'
expected='c5092f3075ccf472b0ea3179837663a558103bd0d0406124361495ab583e9d26'
if not path.exists():
    response=requests.get('https://github.com/KyeongwooLee/AI-Endoscope/releases/download/gsm8k-train-features-v1/gsm8k-train-all-v1.zip',stream=True,timeout=90)
    response.raise_for_status()
    temporary=path.with_suffix('.zip.partial')
    with temporary.open('wb') as stream:
        for chunk in response.iter_content(4*1024*1024): stream.write(chunk)
    os.replace(temporary,path)
digest=hashlib.sha256()
with path.open('rb') as stream:
    for chunk in iter(lambda:stream.read(4*1024*1024),b''): digest.update(chunk)
assert digest.hexdigest()==expected
questions={r['question_id']:r for r in json.loads((root/'input/questions.json').read_text()) if r['dataset']=='gsm8k'}
pairs=set(json.loads((root/'input/existing_pairs.json').read_text()))
count=0
with zipfile.ZipFile(path) as archive:
    names=[name for name in archive.namelist() if 'generation' in name and name.endswith('.jsonl')]
    assert names,'No generation files in frozen baseline release'
    for name in names:
        for line in archive.read(name).decode('utf-8').splitlines():
            r=json.loads(line)
            count+=1
            if r['id'] not in questions: continue
            row=questions[r['id']]
            pairs.add(row['question_group']+':'+hashlib.sha256(r['generation']['text'].encode()).hexdigest())
assert count==7473,count
temporary=root/'input/existing_pairs.json.partial'
temporary.write_text(json.dumps(sorted(pairs),indent=2),encoding='utf-8')
os.replace(temporary,root/'input/existing_pairs.json')
(root/'baseline_dedup.json').write_text(json.dumps({'old_greedy_rows':count,'baseline_zip_sha256':expected,'total_known_pairs':len(pairs)},indent=2))
print('BASELINE_DEDUP_READY',len(pairs),flush=True)
