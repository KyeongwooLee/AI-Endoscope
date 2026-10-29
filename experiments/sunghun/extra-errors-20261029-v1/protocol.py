"""Immutable experiment output and verification helpers (standard library only)."""
import hashlib
import json
import os
from pathlib import Path
import zipfile

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(4*1024*1024),b''): digest.update(data)
    return digest.hexdigest()

def write_json(path,data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_suffix(path.suffix+'.partial')
    with partial.open('w',encoding='utf-8') as stream:
        json.dump(data,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial,path)

def commit(root,name,members,metadata):
    out=Path(root)/'outbox'
    out.mkdir(exist_ok=True)
    destination=out/(name+'.zip')
    if destination.exists(): raise FileExistsError(destination)
    partial=destination.with_suffix('.zip.partial')
    hashes={}
    with zipfile.ZipFile(partial,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for key,data in members.items():
            archive.writestr(key,data)
            hashes[key]=hashlib.sha256(data).hexdigest()
    os.replace(partial,destination)
    receipt={'name':name,'archive_sha256':sha(destination),'bytes':destination.stat().st_size,
             'members':hashes,**metadata}
    verify(destination,receipt)
    write_json(out/(name+'.json'),receipt)
    return receipt

def verify(path,receipt):
    path=Path(path)
    assert path.stat().st_size==receipt['bytes'] and sha(path)==receipt['archive_sha256'],str(path)
    with zipfile.ZipFile(path) as archive:
        assert set(archive.namelist())==set(receipt['members'])
        for member,digest in receipt['members'].items():
            assert hashlib.sha256(archive.read(member)).hexdigest()==digest,member

def read_records(root):
    receipts=sorted((Path(root)/'outbox').glob('generations_*.json'))
    for path in receipts:
        receipt=json.loads(path.read_text(encoding='utf-8'))
        archive=path.with_suffix('.zip')
        verify(archive,receipt)
        with zipfile.ZipFile(archive) as source:
            for line in source.read('records.jsonl').decode('utf-8').splitlines():
                yield json.loads(line)
