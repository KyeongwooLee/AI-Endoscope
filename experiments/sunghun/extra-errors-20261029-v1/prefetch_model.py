import hashlib
import json
import os
from pathlib import Path
import requests

root = Path('/root/AI-Endoscope/extra_errors_20261029')
model_id='Qwen/Qwen2.5-1.5B-Instruct'
revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
destination=root/'model'
destination.mkdir(exist_ok=True)
metadata=requests.get(f'https://huggingface.co/api/models/{model_id}/revision/{revision}?blobs=true',timeout=60)
metadata.raise_for_status()
metadata=metadata.json()
assert metadata['sha']==revision
(root/'model_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
needed={'config.json','generation_config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','special_tokens_map.json','LICENSE'}
files=[item for item in metadata['siblings'] if item['rfilename'] in needed or item['rfilename'].endswith('.safetensors') or item['rfilename'].endswith('.safetensors.index.json')]
manifest=[]
for item in files:
    name=item['rfilename']
    path=destination/name
    expected=item.get('lfs',{}).get('sha256')
    if not path.exists():
        partial=path.with_suffix(path.suffix+'.partial')
        response=requests.get(f'https://huggingface.co/{model_id}/resolve/{revision}/{name}',stream=True,timeout=90)
        response.raise_for_status()
        path.parent.mkdir(parents=True,exist_ok=True)
        with partial.open('wb') as stream:
            for chunk in response.iter_content(4*1024*1024):
                stream.write(chunk)
        os.replace(partial,path)
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''): digest.update(chunk)
    assert not expected or digest.hexdigest()==expected,name
    manifest.append({'file':name,'bytes':path.stat().st_size,'sha256':digest.hexdigest(),'lfs_sha256':expected})
    print('VERIFIED',name,path.stat().st_size,flush=True)
(root/'model_files.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('MODEL_READY',revision,flush=True)
