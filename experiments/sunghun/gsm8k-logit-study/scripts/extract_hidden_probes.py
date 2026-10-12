"""Replay fixed prefixes; store selected raw residuals and full-depth scalar changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM

CHECKPOINTS = (0, 32, 64, 128)
LAYERS = (7, 10, 13, 14)
REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
METRICS = ('previous_norm', 'hidden_norm', 'delta_norm', 'relative_delta', 'cosine_distance')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    temp = Path(str(path) + '.partial')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def residual_metrics(states):
    before, after = states[:-1], states[1:]
    previous = before.norm(dim=-1)
    current = after.norm(dim=-1)
    delta = (after - before).norm(dim=-1)
    relative = delta / previous.clamp_min(1e-12)
    cosine = 1 - ((before * after).sum(-1) / (previous * current).clamp_min(1e-12)).clamp(-1, 1)
    return torch.stack((previous, current, delta, relative, cosine), -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in Path(args.input).read_text().splitlines()]
    if args.limit:
        records = records[:args.limit]
    config = {'input_sha256': sha(args.input), 'script_sha256': sha(__file__),
              'model': 'Qwen/Qwen2.5-1.5B-Instruct', 'revision': REVISION,
              'dtype': 'float32', 'checkpoints': CHECKPOINTS, 'stored_layers': LAYERS,
              'metric_names': METRICS, 'limit': args.limit,
              'hidden_definition': 'Raw decoder-block outputs, BEFORE final RMSNorm; h0=embedding.',
              'position_definition': 'prompt_length-1+t; input contains prompt plus tokens 0..t-1.',
              'torch': torch.__version__, 'transformers': transformers.__version__,
              'python': platform.python_version(), 'attention': 'sdpa', 'tf32': False}
    config = json.loads(json.dumps(config))
    config_path = output / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Extraction configuration mismatch; use a new output directory')
    write_json(config_path, config)
    assert torch.cuda.is_available(), 'GPU required'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(config['model'], revision=REVISION,
        torch_dtype=torch.float32, attn_implementation='sdpa').to('cuda').eval()
    assert len(model.model.layers) == 28
    assert model.config.hidden_size == 1536
    captured, positions = {}, None

    def hook(layer):
        def collect(module, inputs, result):
            hidden = result[0] if isinstance(result, tuple) else result
            captured[layer] = hidden[0, positions].detach().clone()
        return collect

    handles = [model.model.embed_tokens.register_forward_hook(hook(0))]
    handles += [layer.register_forward_hook(hook(index)) for index, layer in enumerate(model.model.layers, 1)]
    gate_matches, gate_total, causal_error = 0, 0, None
    ledger = []
    try:
        with torch.inference_mode():
            for batch_start in range(0, len(records), 32):
                batch_number = batch_start // 32
                file = output / f'batch_{batch_number:04d}.npz'
                receipt = output / f'batch_{batch_number:04d}.json'
                if receipt.exists():
                    saved = json.loads(receipt.read_text())
                    assert sha(file) == saved['sha256']
                    ledger.append(saved)
                    gate_matches += saved['gate_matches']
                    gate_total += saved['gate_total']
                    if saved['causal_error'] is not None:
                        causal_error = saved['causal_error']
                    continue
                ids, phases, targets, times, vectors, curves = [], [], [], [], [], []
                local_matches, local_total, local_causal = 0, 0, None
                for record in records[batch_start:batch_start + 32]:
                    checkpoints = [t for t in CHECKPOINTS if t < len(record['generated_token_ids'])]
                    assert checkpoints
                    prompt = record['prompt_token_ids']
                    token_ids = prompt + record['generated_token_ids'][:max(checkpoints)]
                    positions = torch.tensor([len(prompt) - 1 + t for t in checkpoints], device='cuda')
                    captured.clear()
                    model.model(input_ids=torch.tensor([token_ids], device='cuda'), use_cache=False)
                    assert sorted(captured) == list(range(29))
                    states = torch.stack([captured[layer] for layer in range(29)])
                    assert torch.isfinite(states).all()
                    selected = states[list(LAYERS)].permute(1, 0, 2).cpu().numpy()
                    metrics = residual_metrics(states).permute(1, 0, 2).cpu().numpy()
                    # Quality diagnostics only: chosen next token never enters probe inputs.
                    if batch_start == 0:
                        predicted = model.lm_head(model.model.norm(states[-1])).argmax(-1).cpu().tolist()
                        actual = [record['generated_token_ids'][t] for t in checkpoints]
                        local_matches += sum(a == b for a, b in zip(predicted, actual))
                        local_total += len(actual)
                    if batch_start == 0 and local_causal is None and 32 in checkpoints and max(checkpoints) > 32:
                        reference = states[14, checkpoints.index(32)].clone()
                        positions = torch.tensor([len(prompt) - 1 + 32], device='cuda')
                        captured.clear()
                        model.model(input_ids=torch.tensor([prompt + record['generated_token_ids'][:32]], device='cuda'), use_cache=False)
                        local_causal = float((captured[14][0] - reference).abs().max().item())
                        if not torch.allclose(captured[14][0], reference, atol=3e-4, rtol=3e-4):
                            raise ValueError('Prefix truncation causality check failed')
                    for index, t in enumerate(checkpoints):
                        ids.append(record['id'])
                        phases.append(record['phase'])
                        targets.append(record['incorrect'])
                        times.append(t)
                        vectors.append(selected[index])
                        curves.append(metrics[index])
                    del states
                temp = Path(str(file) + '.partial')
                with temp.open('wb') as stream:
                    np.savez_compressed(stream, ids=np.asarray(ids), phases=np.asarray(phases),
                        y=np.asarray(targets, dtype=np.int8), checkpoints=np.asarray(times, dtype=np.int16),
                        hidden=np.stack(vectors), trajectory=np.stack(curves))
                os.replace(temp, file)
                saved = {'file': file.name, 'sha256': sha(file), 'bytes': file.stat().st_size,
                    'problems': len(records[batch_start:batch_start + 32]), 'rows': len(ids),
                    'gate_matches': local_matches, 'gate_total': local_total, 'causal_error': local_causal}
                write_json(receipt, saved)
                ledger.append(saved)
                gate_matches += local_matches
                gate_total += local_total
                if local_causal is not None:
                    causal_error = local_causal
                elapsed = time.monotonic() - started
                done = batch_start + saved['problems']
                write_json(output / 'progress.json', {'problems': done, 'total': len(records),
                    'elapsed_seconds': elapsed, 'remaining_seconds_estimate': elapsed / done * (len(records) - done)})
                print(f'Extracted {done}/{len(records)}; elapsed {elapsed:.1f}s; GPU peak {torch.cuda.max_memory_allocated()/2**30:.2f}GiB', flush=True)
                if batch_start == 0 and gate_matches / gate_total < .99:
                    raise ValueError(f'Generation replay gate failed: {gate_matches}/{gate_total}')
    finally:
        for handle in handles:
            handle.remove()
    write_json(output / 'complete.json', {'batches': ledger, 'problems': len(records),
        'elapsed_seconds': time.monotonic() - started, 'causal_max_abs_error': causal_error,
        'replay_argmax_agreement': gate_matches / gate_total,
        'peak_gpu_gib': torch.cuda.max_memory_allocated() / 2**30})
    print('EXTRACTION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
