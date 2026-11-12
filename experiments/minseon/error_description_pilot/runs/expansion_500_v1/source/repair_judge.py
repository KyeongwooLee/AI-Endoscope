import os
os.environ['USE_TF']='0';os.environ['USE_FLAX']='0';os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import json,torch,shutil
from transformers import AutoTokenizer,AutoModelForCausalLM
out=Path(__file__).resolve().parents[1];f=out/'ai_pair_judgments.jsonl';rows=[json.loads(l) for l in f.read_text().splitlines()];missing=[r for r in rows if r['judgment'] is None]
if not missing:raise SystemExit(0)
backup=out/'ai_pair_judgments.before_format_repair.jsonl'
if not backup.exists():shutil.copyfile(f,backup)
pack={r['pair_id']:r['prompt'] for r in [json.loads(l) for l in (out/'judge_packets.jsonl').read_text().splitlines()]};path='/home/coder/yaicon-errorprobe/models/Qwen3-4B-Instruct-2507';torch.set_num_threads(4);torch.manual_seed(20261112)
tok=AutoTokenizer.from_pretrained(path,local_files_only=True,padding_side='left');model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval()
def gen(ps,cap):
 txt=[tok.apply_chat_template([{'role':'user','content':p}],tokenize=False,add_generation_prompt=True) for p in ps];inp=tok(txt,padding=True,return_tensors='pt').to('cuda')
 with torch.inference_mode():res=model.generate(**inp,max_new_tokens=cap,do_sample=False,pad_token_id=tok.pad_token_id)
 return tok.batch_decode(res[:,inp['input_ids'].shape[-1]:],skip_special_tokens=True)
def parse(raw):
 s=raw.strip()
 if s.startswith('```'):s=s.split('\n',1)[1].rsplit('```',1)[0]
 z=json.loads(s);assert z['judgment'] in ['same','different','uncertain'] and isinstance(z['reason'],str) and z['reason'];return z
for cap in [512,1024]:
 todo=[r for r in missing if r['judgment'] is None]
 for i in range(0,len(todo),4):
  batch=todo[i:i+4];texts=gen([pack[r['pair_id']] for r in batch],cap)
  for r,raw in zip(batch,texts):
   r['raw_attempts'].append(raw);r.setdefault('format_repair_token_caps',[]).append(cap)
   try:r.update(parse(raw))
   except Exception:pass
 tmp=f.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));tmp.replace(f)
 print('FORMAT REPAIR cap',cap,'remaining',sum(r['judgment'] is None for r in rows),flush=True)
 if all(r['judgment'] is not None for r in rows):break
