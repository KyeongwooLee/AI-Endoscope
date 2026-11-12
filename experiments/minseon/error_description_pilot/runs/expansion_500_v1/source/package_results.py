from pathlib import Path
import json,hashlib,zipfile
out=Path(__file__).resolve().parents[1]
assert json.load(open(out/'validation.json'))['passed']
assert json.load(open(out/'analysis_status.json'))['report_complete']
files=[p for p in sorted(out.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact_manifest.json']
manifest={'date_kst':'2026-11-12','status':'completed_AI_exploratory_experiment','files':{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(out/'artifact_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));files.append(out/'artifact_manifest.json')
archive=out.parent/(out.name+'_results.zip')
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in files:z.write(p,arcname=out.name+'/'+str(p.relative_to(out)))
h=hashlib.sha256(archive.read_bytes()).hexdigest();archive.with_suffix('.zip.sha256').write_text(h+'  '+archive.name+'\n')
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'files':len(files),'sha256':h}))
