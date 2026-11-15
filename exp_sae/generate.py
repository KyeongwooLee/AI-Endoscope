import time
from pathlib import Path

from sae_error_clustering.tokens import label_answer
from .common import (digest, environment, lock, object_hash, prepared, read_json,
                     read_problem_responses, response_path, rid, runtime, sample_seed,
                     save, space, write_rows)


SYSTEM = 'Solve the math problem step by step. End with a final line exactly in the form #### <number>.'


def run(cfg, worker, limit=None):
    if not 0 <= worker < cfg['workers'] or limit is not None and limit <= 0:
        raise ValueError('Invalid worker or problem limit')
    environment(cfg)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    problems, manifest = prepared(cfg)
    out = Path(cfg['run_dir'])
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 GPU required; select one GPU with CUDA_VISIBLE_DEVICES')
    selected = [p for p in problems if p['pid'] % cfg['workers'] == worker][:limit]
    with lock(out / f'generate-worker-{worker}'):
        signature = {'runtime': runtime(), 'config_sha256': object_hash(cfg),
                     'dataset_revision': manifest['dataset_revision'],
                     'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda}
        state = out / f'generation-worker-{worker}.json'
        if state.exists() and read_json(state) != signature:
            raise ValueError('Generation runtime changed; use a new run')
        save(state, signature)
        pending = []
        for problem in selected:
            path = response_path(cfg, problem['pid'])
            if path.exists() or path.with_suffix('.meta.json').exists():
                read_problem_responses(cfg, problem)
            else:
                pending.append(problem)
        if not pending:
            print(f'worker={worker}: all requested problem bundles verified; no model load')
            return
        space(cfg, 5 * 2**30)
        tokenizer = AutoTokenizer.from_pretrained(cfg['model'], revision=cfg['tokenizer_revision'])
        model = AutoModelForCausalLM.from_pretrained(cfg['model'], revision=cfg['model_revision'],
            torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda:0').eval()
        model.requires_grad_(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        stops = model.generation_config.eos_token_id
        stops = [stops] if isinstance(stops, int) else list(stops)
        for problem in pending:
            target = response_path(cfg, problem['pid'])
            if target.exists() or target.with_suffix('.meta.json').exists():
                read_problem_responses(cfg, problem)
                continue
            space(cfg, 16 * 2**20)
            prompt = tokenizer.apply_chat_template([{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': problem['question']}], tokenize=True,
                add_generation_prompt=True, return_dict=False)
            if len(prompt) + cfg['generation']['max_new_tokens'] > model.config.max_position_embeddings:
                raise ValueError('Context exceeds model limit; never truncate input')
            records, seen = [], {}
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            for repeat in range(cfg['repeats']):
                seed = sample_seed(problem['pid'], repeat, cfg['seed'])
                set_seed(seed)
                with torch.inference_mode():
                    output = model.generate(input_ids=torch.tensor([prompt], device='cuda:0'),
                        attention_mask=torch.ones((1, len(prompt)), dtype=torch.long, device='cuda:0'),
                        do_sample=True, **cfg['generation'], eos_token_id=stops,
                        pad_token_id=tokenizer.pad_token_id or stops[0])[0, len(prompt):].tolist()
                if not output:
                    raise ValueError('Empty generated token sequence')
                finish = 'stop' if output[-1] in stops else 'length'
                text = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                response_id = rid(problem['pid'], repeat)
                canonical = seen.setdefault(text, response_id)
                records.append({**problem, 'id': response_id, 'repeat': repeat,
                    'response': text, 'canonical_id': canonical, 'prompt_token_ids': prompt,
                    'output_token_ids': output, 'seed': seed, 'finish': finish,
                    'generation': cfg['generation'], 'system_prompt': SYSTEM,
                    'model_revision': cfg['model_revision'], 'tokenizer_revision': cfg['tokenizer_revision'],
                    **label_answer(text, problem['gold'], finish == 'length')})
            write_rows(target, records)
            save(target.with_suffix('.meta.json'), {'sha256': digest(target),
                'config_sha256': object_hash(cfg), 'runtime_sha256': object_hash(signature),
                'seconds': time.monotonic() - started, 'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
                'bundle_bytes': target.stat().st_size, 'disk_free_bytes': space(cfg)})
            print(f'worker={worker} pid={problem["pid"]} responses={len(records)}', flush=True)
