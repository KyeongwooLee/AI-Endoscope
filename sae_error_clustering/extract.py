"""GPU-only extraction. No model import/download occurs at package import time."""
from pathlib import Path

from .common import (atomic_json, atomic_npz, digest, object_hash, paths, prepared,
                     read_json, runtime_signature)
from .tokens import parse_answer, token_regions


def reasoning_tokens(tokenizer, row):
    text, regions = token_regions(tokenizer, row['output_token_ids'])
    if text != row['response']:
        raise ValueError('Decoded text differs from stored response')
    if row['finish'] == 'length':
        raise ValueError('truncated_completion')
    if parse_answer(text)['status'] != 'ok':
        raise ValueError('unparsed_final_line')
    indices = [r['token_index'] for r in regions if r['region'] == 'reasoning'
               and r['included'] and not r['boundary_overlap']]
    if not indices:
        raise ValueError('empty_reasoning')
    return indices, [(r['char_start'], r['char_end']) for r in regions]


def encode(h, weights):
    return ((h - weights['b_dec']) @ weights['W_enc'] + weights['b_enc']).relu()


def decode(z, weights):
    return z @ weights['W_dec'] + weights['b_dec']


def hidden_chunks(backbone, layer, prompt, output, chunk_size, device):
    import torch
    cache = backbone(input_ids=torch.tensor([prompt], device=device), use_cache=True).past_key_values
    captured = {}

    def hook(_module, _args, value):
        captured['hidden'] = value[0] if isinstance(value, tuple) else value

    handle = backbone.layers[layer].register_forward_hook(hook)
    try:
        for start in range(0, len(output), chunk_size):
            ids = output[start:start + chunk_size]
            result = backbone(input_ids=torch.tensor([ids], device=device),
                              past_key_values=cache, use_cache=True)
            cache = result.past_key_values
            yield start, captured.pop('hidden')[0]
    finally:
        handle.remove()


def aggregate(chunks, selected, width, hidden_size, weights):
    import torch
    selected = set(selected)
    device = weights['b_dec'].device
    total_z = torch.zeros(width, device=device)
    total_h = torch.zeros(hidden_size, device=device)
    maxima = torch.full((width,), -float('inf'), device=device)
    peaks = torch.full((width,), -1, dtype=torch.int32, device=device)
    error = energy = 0.0
    count = 0
    for start, hidden in chunks:
        local = [i for i in range(len(hidden)) if start + i in selected]
        if not local:
            continue
        h = hidden[local].float()
        z = encode(h, weights)
        reconstructed = decode(z, weights)
        if not all(torch.isfinite(a).all().item() for a in [h, z, reconstructed]):
            raise ValueError('Non-finite activation or SAE reconstruction')
        total_z += z.sum(0)
        total_h += h.sum(0)
        values, positions = z.max(0)
        source = torch.tensor(local, device=device, dtype=torch.int32)[positions] + start
        peaks = torch.where(values > maxima, source, peaks)
        maxima = torch.maximum(maxima, values)
        error += (reconstructed - h).square().sum().item()
        energy += h.square().sum().item()
        count += len(h)
    if count != len(selected) or count == 0:
        raise ValueError('Missing/empty reasoning token coverage')
    peaks[maxima <= 0] = -1
    return ((total_z / count).cpu().numpy(), (total_h / count).cpu().numpy(), peaks.cpu().numpy(),
            {'tokens': count, 'reconstruction_mse': error / (count * hidden_size),
             'relative_squared_error': error / max(energy, 1e-12),
             'active_features': int((total_z > 0).sum().item())})


def read_shard(path, expected_ids, signature):
    import numpy as np
    meta = read_json(path.with_suffix('.json'))
    if meta['signature'] != signature or meta['sha256'] != digest(path):
        raise ValueError(f'Corrupt/incompatible shard: {path}')
    with np.load(path, allow_pickle=False) as arrays:
        if arrays['ids'].tolist() != expected_ids or [d['id'] for d in meta['rows']] != expected_ids:
            raise ValueError('Shard ID order mismatch')
        n = len(expected_ids)
        for key, shape in [('sae_mean', (n, 24576)), ('hidden_mean', (n, 1536)), ('peaks', (n, 24576))]:
            if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
                raise ValueError(f'Invalid shard array: {key}')
        return {k: arrays[k].copy() for k in arrays.files}, meta


def run(cfg, limit=None):
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows, manifest = prepared(cfg)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A CUDA GPU with BF16 support is required; no silent precision fallback')
    _, out = paths(cfg)
    runtime = runtime_signature(cfg, manifest)
    runtime['gpu'] = {'name': torch.cuda.get_device_name(), 'cuda': torch.version.cuda,
                        'capability': list(torch.cuda.get_device_capability())}
    signature = object_hash(runtime)
    run_path = out / 'extraction.json'
    if run_path.exists() and read_json(run_path)['signature'] != signature:
        raise ValueError('Extraction code, environment, config or input changed; use a new output directory')
    was_complete = run_path.exists() and read_json(run_path).get('complete', False)
    atomic_json(out / 'runtime.json', runtime)
    atomic_json(run_path, {'signature': signature, 'complete': False})
    kwargs = dict(repo_id=cfg['sae'], revision=cfg['sae_revision'], subfolder=cfg['sae_subfolder'])
    sae_config = read_json(hf_hub_download(filename='cfg.json', **kwargs))
    if (sae_config.get('architecture') != 'standard' or sae_config.get('d_in') != 1536
            or sae_config.get('d_sae') != 24576 or sae_config.get('normalize_activations') != 'none'
            or sae_config.get('reshape_activations') != 'none' or not sae_config.get('apply_b_dec_to_input')
            or sae_config['metadata']['hook_name'] != f'model.layers.{cfg["layer"]}'
            or sae_config['metadata']['model_name'] != cfg['model']):
        raise ValueError('Unsupported SAE architecture, model or hook')
    weight_path = hf_hub_download(filename='sae_weights.safetensors', **kwargs)
    weights = load_file(weight_path, device='cuda')
    expected = {'W_enc': (1536, 24576), 'W_dec': (24576, 1536), 'b_enc': (24576,), 'b_dec': (1536,)}
    if set(weights) != set(expected) or any(tuple(weights[k].shape) != s for k, s in expected.items()):
        raise ValueError('SAE weight shape mismatch')
    weights = {k: v.float() for k, v in weights.items()}
    atomic_json(out / 'sae_provenance.json', {'config': sae_config, 'weights_sha256': digest(weight_path),
               'model_revision_match': 'SAE metadata does not specify a base-model revision; unverified'})
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(cfg['model'], revision=cfg['model_revision'])
    model = AutoModelForCausalLM.from_pretrained(cfg['model'], revision=cfg['model_revision'],
        torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    if len(model.model.layers) != 28 or model.config.hidden_size != 1536:
        raise ValueError('Unexpected Qwen architecture')
    shards = out / 'shards'
    shards.mkdir(exist_ok=True)
    stop = min(limit or len(rows), len(rows))
    with torch.inference_mode():
        for start in range(0, stop, cfg['shard_size']):
            batch = rows[start:min(start + cfg['shard_size'], len(rows))]
            path = shards / f'{start:06d}.npz'
            ids = [r['id'] for r in batch]
            if path.exists() and path.with_suffix('.json').exists():
                read_shard(path, ids, signature)
                continue
            means = np.zeros((len(batch), 24576), dtype=np.float32)
            hidden = np.zeros((len(batch), 1536), dtype=np.float32)
            peaks = np.full((len(batch), 24576), -1, dtype=np.int32)
            details = []
            for i, row in enumerate(batch):
                try:
                    selected, offsets = reasoning_tokens(tokenizer, row)
                except ValueError as exc:
                    details.append({'id': row['id'], 'status': 'excluded', 'reason': str(exc)})
                    continue
                if len(row['prompt_token_ids']) + len(row['output_token_ids']) > model.config.max_position_embeddings:
                    raise ValueError(f'Context too long; never truncate input: {row["id"]}')
                chunks = hidden_chunks(model.model, cfg['layer'], row['prompt_token_ids'],
                                       row['output_token_ids'], cfg['chunk_size'], 'cuda')
                means[i], hidden[i], peaks[i], metrics = aggregate(chunks, selected, 24576, 1536, weights)
                details.append({'id': row['id'], 'status': 'ok', 'offsets': offsets, **metrics})
            atomic_npz(path, ids=np.asarray(ids), sae_mean=means, hidden_mean=hidden, peaks=peaks)
            atomic_json(path.with_suffix('.json'), {'signature': signature, 'sha256': digest(path), 'rows': details})
            print(f'Committed {start + len(batch)}/{len(rows)} records', flush=True)
    atomic_json(run_path, {'signature': signature, 'complete': was_complete or stop == len(rows), 'processed': stop})


def load_features(cfg):
    import numpy as np
    rows, _ = prepared(cfg)
    _, out = paths(cfg)
    run = read_json(out / 'extraction.json')
    if not run['complete']:
        raise ValueError('Extraction is incomplete; cluster/report require all records')
    blocks, details = [], []
    for start in range(0, len(rows), cfg['shard_size']):
        batch = rows[start:start + cfg['shard_size']]
        block, meta = read_shard(out / 'shards' / f'{start:06d}.npz', [r['id'] for r in batch], run['signature'])
        blocks.append(block)
        details.extend(meta['rows'])
    return rows, {k: np.concatenate([b[k] for b in blocks]) for k in ['sae_mean', 'hidden_mean', 'peaks']}, details
