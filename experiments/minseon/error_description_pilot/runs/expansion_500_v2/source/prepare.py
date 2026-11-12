from pathlib import Path
import json,hashlib,shutil,sys,copy
out=Path(__file__).resolve().parents[1];old=out.parent/'expansion_500_v1';base=out.parents[1];sys.path.insert(0,str(base))
from edp.annotation import validate_output
from exact_quote import attach_unique_quote_offsets
jl=lambda p:[json.loads(l) for l in p.read_text().splitlines() if l]
def save(n,v): (out/n).write_text(json.dumps(v,ensure_ascii=False,indent=2))
def lines(n,v): (out/n).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in v))
assert not (out/'protocol.json').exists(),'Do not overwrite an existing experiment'
save('baseline_hashes.json',{str(p.relative_to(old)):hashlib.sha256(p.read_bytes()).hexdigest() for p in old.rglob('*') if p.is_file()})
for n in ['sample.jsonl','config.json','annotation_audit_original_diagnoses.json','pair_audit_codex.json','source_attribution']:
 p=old/n
 if p.is_dir():shutil.copytree(p,out/n)
 else:shutil.copy2(p,out/n)
for n in ['annotation','packets']:(out/n).mkdir(exist_ok=True)
for c in 'EG':shutil.copy2(old/f'packets/{c}.jsonl',out/f'packets/{c}.jsonl')
shutil.copy2(old/'annotation/E.jsonl',out/'annotation/E.jsonl')
rows=jl(old/'annotation/G.jsonl');samples={r['id']:r for r in jl(out/'sample.jsonl')};pack={r['source_id']:r for r in jl(out/'packets/G.jsonl')};cfg=json.load(open(out/'config.json'));repairs=[]
for r in rows:
 o=r['output'];sid=r['source_id']
 if r['valid'] or not isinstance(o,dict) or o.get('status')!='supported_error':continue
 quote=o.get('evidence_quote');s=samples[sid]['response']
 if not isinstance(quote,str) or not quote or s.count(quote)!=1:continue
 try:fixed=attach_unique_quote_offsets(o,s)
 except ValueError:continue
 errors,warnings,overlay=validate_output('G',fixed,pack[sid],cfg)
 if errors:continue
 assert fixed['error_description']==o['error_description']
 repairs.append({'source_id':sid,'old_errors':r['errors'],'old_start':o.get('evidence_start'),'old_end':o.get('evidence_end'),'new_start':fixed['evidence_start'],'new_end':fixed['evidence_end'],'evidence_quote':quote,'description_unchanged':True})
 r.update(output=fixed,valid=True,errors=[],warnings=warnings,overlay=overlay,postprocess='unique exact quote offset computed by code; description unchanged')
lines('annotation/G.jsonl',rows);save('offset_repairs.json',repairs)
assert len(repairs)==77
(out/'embeddings').mkdir(exist_ok=True)
for c in 'REQ':
 for suffix in ['.npz','_inputs.jsonl']:shutil.copy2(old/f'embeddings/{c}{suffix}',out/f'embeddings/{c}{suffix}')
shutil.copy2(old/'source/analyze.py',out/'source/analyze.py')
cfg.update(protocol_name='error-description-expansion-500-quality-followup',protocol_version='expansion500-v2',run_dir='runs/expansion_500_v2');save('config.json',cfg)
save('protocol.json',{'baseline':'expansion_500_v1','sample_unchanged':True,'intervention':'unique exact quote offsets only; no description or source text modifications','repair_n':77,'clustering':'same settings; full and eligible common cohorts; also compare on fixed baseline common cohort','reference_screening':'fresh Qwen3-4B diagnosis without reference for all500; provisional triage, not truth; assess on existing20 AI audits','judge_validation':'original vs revised reference-free structured prompt on frozen synthetic positive/negative/uncertain controls with swapped order; existing10 real pair AI diagnoses used as additional checks','no_human_ground_truth':True,'gate':'no new full-retrieval accuracy claim unless evaluator trustworthy; synthetic success alone insufficient','main_embedding':'only existing E/G descriptions; reference-free screening text NOT substituted into primary vectors'})
print('prepared',len(repairs),'exact quote repairs')
