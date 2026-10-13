"""Pilot-only batch throughput measurement, never used for accuracy tuning."""
import time
from collect import Engine,read,dump,RUN,REBUILD,REVIEW

def main():
 e=Engine();rows=[read(p) for p in sorted((RUN/'records').glob('*.json')) if read(p)['split']=='pilot'][:16];result={}
 for size in [16]:
  try:
   e.torch.cuda.reset_peak_memory_stats();st=time.perf_counter()
   originals=e.generate_batch([e.chat(e.cfg['user_instruction']+r['question']) for r in rows[:size]],512)
   prompts=[]
   for r in rows[:size]:prompts.extend([e.chat(REBUILD+r['question']),e.chat(REVIEW+'Question:\n'+r['question']+'\n\nProposed response:\n'+r['original']['decoded_output'])])
   repairs=e.generate_batch(prompts,512)
   result[str(size)]=dict(seconds=time.perf_counter()-st,original_seconds=sum(o['seconds'] for o in originals),repair_seconds=sum(o['seconds'] for o in repairs),peak_reserved=e.torch.cuda.max_memory_reserved(),same_original_text=sum(o['decoded_output']==r['original']['decoded_output'] for r,o in zip(rows,originals)),n=size,outputs=dict(originals=originals,repairs=repairs))
  except e.torch.cuda.OutOfMemoryError as exc:result[str(size)]={'oom':str(exc)};e.torch.cuda.empty_cache()
 dump(RUN/'batch_benchmark.json',result)
 print({k:{a:v for a,v in r.items() if a!='outputs'} for k,r in result.items()},flush=True)
if __name__=='__main__':main()
