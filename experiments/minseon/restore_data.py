"""Restore published run artifacts; verify every part/member before writing."""
import argparse,hashlib,json,tempfile,zipfile
from pathlib import Path

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'project');parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
 base=Path(__file__).resolve().parent;manifest=json.loads((base/'publication_manifest.json').read_text());target=args.output.resolve();count=0
 for a in manifest['archives']:
  with tempfile.TemporaryDirectory(prefix='endoscope_restore_') as temp:
   archive=Path(temp)/a['name']
   with archive.open('wb') as f:
    for part in a['parts']:
     p=base/part['path'];assert p.stat().st_size==part['bytes'];assert sha(p)==part['sha256'],p;f.write(p.read_bytes())
   assert sha(archive)==a['sha256']
   with zipfile.ZipFile(archive) as z:
    assert set(z.namelist())==set(a['members'])
    for name,h in a['members'].items():
     dest=(target/name).resolve();assert dest.is_relative_to(target),name
     data=z.read(name);assert hashlib.sha256(data).hexdigest()==h,name
     if not args.verify_only:
      if dest.exists():assert sha(dest)==h,f'Refusing to overwrite changed file: {dest}'
      else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
     count+=1
  print(('Verified' if args.verify_only else 'Restored'),a['name'],flush=True)
 print(f'{count} file contents verified; original local paths and historical hash manifests are preserved.')
if __name__=='__main__':main()
