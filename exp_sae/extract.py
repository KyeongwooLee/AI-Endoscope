import time
from pathlib import Path

from sae_error_clustering.extract import hidden_chunks, reasoning_tokens
from . import label, sae
from .common import (atomic_npz, digest, environment, lock, object_hash, prepared,
                     read_json, read_problem_responses, response_path, rows, runtime,
                     save, space, write_rows)


def aggregate(hidden, selected, weights, entry, chunk_size):
    import numpy as np
    import torch
    if len(hidden) != len(selected) or len(set(selected)) != len(selected) or not len(hidden):
        raise ValueError('Hidden/reasoning token alignment mismatch')
    width, d = entry['d_sae'], entry['d_in']
    total = torch.zeros(width, device=weights['b_dec'].device)
    maxima = torch.zeros_like(total)
    counts = torch.zeros_like(total)
    peaks = torch.full((width,), -1, dtype=torch.int32, device=total.device)
    error = energy = l0 = 0.
    for start in range(0, len(hidden), chunk_size):
        h = torch.as_tensor(hidden[start:start + chunk_size], device=total.device).float()
        z, reconstruction = sae.encode(h, weights, entry)
        if not all(torch.isfinite(t).all().item() for t in (h, z, reconstruction)):
            raise ValueError('Non-finite hidden/SAE features')
        total += z.sum(0)
        counts += (z > 0).sum(0)
        values, positions = z.max(0)
        indices = torch.tensor(selected[start:start + len(h)], dtype=torch.int32, device=total.device)[positions]
        peaks = torch.where(values > maxima, indices, peaks)
        maxima = torch.maximum(maxima, values)
        error += (h - reconstruction).square().sum().item()
        energy += h.square().sum().item()
        l0 += (z > 0).sum().item()
    n = len(hidden)
    features = {'sae_mean': (total / n).cpu().numpy(), 'sae_max': maxima.cpu().numpy(),
        'sae_frequency': (counts / n).cpu().numpy(), 'peaks': peaks.cpu().numpy(),
        'hidden_mean': np.asarray(hidden, dtype=np.float32).mean(0)}
    metrics = {'tokens': n, 'reconstruction_mse': error / (n * d),
        'relative_squared_error': error / max(energy, 1e-12),
        'mean_l0': l0 / n, 'active_fraction': l0 / (n * width)}
    return features, metrics


def read_hidden(path, row, entry, selected):
    import numpy as np
    meta = read_json(path.with_suffix('.json'))
    if (meta['sha256'] != digest(path) or meta['response_sha256'] != object_hash(row)
            or meta['layer'] != entry['layer'] or meta['selected'] != selected):
        raise ValueError('Temporary hidden hash/token identity mismatch')
    with np.load(path, allow_pickle=False) as data:
        if data['hidden'].shape != (len(selected), entry['d_in']) or not np.isfinite(data['hidden']).all():
            raise ValueError('Temporary hidden shape/non-finite mismatch')
        return data['hidden'].copy()


def verify_bundle(base, ids, signature, width, hidden_size, feature_ids):
    import numpy as np
    meta = read_json(base.with_suffix('.meta.json'))
    quality_path, feature_path = base.with_suffix('.jsonl'), base.with_suffix('.npz')
    if (meta['signature'] != signature or meta['quality_sha256'] != digest(quality_path)
            or meta['feature_sha256'] != digest(feature_path)):
        raise ValueError('Corrupt/incompatible extraction bundle; hidden retained')
    quality = list(rows(quality_path))
    if [r['id'] for r in quality] != ids:
        raise ValueError('Extraction quality ID order mismatch')
    with np.load(feature_path, allow_pickle=False) as data:
        if data['ids'].tolist() != feature_ids:
            raise ValueError('Feature ID order mismatch')
        n = len(feature_ids)
        for key, columns in [('sae_mean', width), ('sae_max', width), ('sae_frequency', width),
                             ('peaks', width), ('hidden_mean', hidden_size)]:
            if data[key].shape != (n, columns) or not np.isfinite(data[key]).all():
                raise ValueError(f'Invalid feature shape/values: {key}')
    return quality


def delete_hidden(path, base, ids, signature, entry, feature_ids):
    import numpy as np
    quality = verify_bundle(base, ids, signature, entry['d_sae'], entry['d_in'], feature_ids)
    meta = read_json(path.with_suffix('.json'))
    committed = {q['id']: q for q in quality if q['status'] == 'ok'}
    if (meta['id'] not in committed or digest(path) != meta['sha256']
            or committed[meta['id']]['hidden_sha256'] != meta['sha256']):
        raise ValueError('Temporary hidden has no verified committed aggregate')
    if path.resolve().parent != (base.parent / 'temporary_hidden').resolve():
        raise ValueError('Hidden deletion outside owned scratch directory')
    with np.load(path, allow_pickle=False) as arrays:
        if (arrays['hidden'].shape != (committed[meta['id']]['tokens'], entry['d_in'])
                or not np.isfinite(arrays['hidden']).all()):
            raise ValueError('Temporary hidden shape/content mismatch before deletion')
    path.unlink()
    path.with_suffix('.json').unlink()


def run(cfg, limit=None, sae_id=None):
    if limit is not None and limit <= 0:
        raise ValueError('Limit must be a positive number of problems')
    environment(cfg)
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    problems, _ = prepared(cfg)
    registry = sae.registry(cfg)
    entries = [e for e in registry['entries'] if e['status'] == 'metadata_verified' and (sae_id is None or e['id'] == sae_id)]
    if not entries:
        raise ValueError('No metadata-verified SAE selected; inspect registry exclusion reasons')
    label_path = Path(cfg['run_dir']) / 'labels' / 'labels.jsonl'
    labels = label.load(cfg) if label_path.exists() else []
    label_ids = {r['id'] for r in labels}
    label_hash = object_hash(labels)
    out = Path(cfg['run_dir']) / 'extraction'
    with lock(out):
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('CUDA BF16 required')
        model = tokenizer = None
        gpu = {'name': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda}
        current = read_json(out / 'index.json') if (out / 'index.json').exists() else {}
        for entry in entries:
            signature = object_hash({'config': cfg, 'runtime': runtime(), 'entry': entry, 'labels': label_hash, 'gpu': gpu})
            directory = out / entry['id'] / label_hash
            directory.mkdir(parents=True, exist_ok=True)
            state_path = directory / 'run.json'
            if state_path.exists() and read_json(state_path)['signature'] != signature:
                raise ValueError('Extraction runtime changed; use a new run')
            previous = read_json(state_path) if state_path.exists() else {}
            if (previous.get('processed_problems', 0) >= len(problems[:limit])
                    and (previous.get('complete') or limit is not None)):
                for problem in problems[:limit]:
                    records = read_problem_responses(cfg, problem)
                    ids = [r['id'] for r in records]
                    base = directory / f'{problem["pid"]:05d}'
                    meta = read_json(base.with_suffix('.meta.json'))
                    bundle_signature = object_hash([signature, digest(response_path(cfg, problem['pid']))])
                    verify_bundle(base, ids, bundle_signature, entry['d_sae'], entry['d_in'], meta['feature_ids'])
                    for row in records:
                        temporary = directory / 'temporary_hidden' / (row['id'] + '.npz')
                        if temporary.exists():
                            delete_hidden(temporary, base, ids, bundle_signature, entry, meta['feature_ids'])
                current[entry['id']] = str(directory.relative_to(out))
                save(out / 'index.json', current)
                print(f'{entry["id"]}: completed bundles verified; no checkpoint download')
                continue
            if model is None:
                space(cfg, 5 * 2**30)
                tokenizer = AutoTokenizer.from_pretrained(cfg['model'], revision=cfg['tokenizer_revision'])
                model = AutoModelForCausalLM.from_pretrained(cfg['model'], revision=cfg['model_revision'],
                    torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda:0').eval()
                model.requires_grad_(False)
                torch.backends.cuda.matmul.allow_tf32 = False
                if model.config.hidden_size != 1536 or len(model.model.layers) != 28:
                    raise ValueError('Unexpected Qwen architecture')
            save(state_path, {'signature': signature, 'entry': entry, 'labels_sha256': label_hash, 'complete': False})
            weights, checkpoint = sae.load(cfg, entry)
            start_time = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            for problem in problems[:limit]:
                records = read_problem_responses(cfg, problem)
                base = directory / f'{problem["pid"]:05d}'
                ids = [r['id'] for r in records]
                temp_dir = directory / 'temporary_hidden'
                temp_dir.mkdir(exist_ok=True)
                bundle_signature = object_hash([signature, digest(response_path(cfg, problem['pid']))])
                if base.with_suffix('.meta.json').exists():
                    meta = read_json(base.with_suffix('.meta.json'))
                    verify_bundle(base, ids, bundle_signature, entry['d_sae'], entry['d_in'], meta['feature_ids'])
                    for row in records:
                        temporary = temp_dir / (row['id'] + '.npz')
                        if temporary.exists():
                            delete_hidden(temporary, base, ids, bundle_signature, entry, meta['feature_ids'])
                    continue
                quality, features, kept = [], [], []
                temporaries = []
                for row in records:
                    detail = {'id': row['id'], 'pid': row['pid'], 'partition': row['partition'],
                        'answer_correct': row['answer_correct'], 'truncated': row['truncated'],
                        'parse_status': row['parse_status'], 'duplicate': row['canonical_id'] != row['id']}
                    try:
                        selected, offsets = reasoning_tokens(tokenizer, row)
                    except ValueError as exc:
                        quality.append({**detail, 'status': 'excluded', 'reason': str(exc)})
                        continue
                    if len(row['prompt_token_ids']) + len(row['output_token_ids']) > model.config.max_position_embeddings:
                        raise ValueError('Context exceeds model limit')
                    temporary = temp_dir / (row['id'] + '.npz')
                    space(cfg, len(selected) * entry['d_in'] * 4 + entry['d_sae'] * 32)
                    if temporary.exists() and temporary.with_suffix('.json').exists():
                        hidden = read_hidden(temporary, row, entry, selected)
                    else:
                        pieces = []
                        chosen = set(selected)
                        with torch.inference_mode():
                            for start, h in hidden_chunks(model.model, entry['layer'], row['prompt_token_ids'],
                                    row['output_token_ids'], cfg['chunk_size'], 'cuda:0'):
                                local = [i for i in range(len(h)) if start + i in chosen]
                                if local:
                                    pieces.append(h[local].float().cpu().numpy())
                        hidden = np.concatenate(pieces)
                        atomic_npz(temporary, hidden=hidden)
                        save(temporary.with_suffix('.json'), {'id': row['id'], 'layer': entry['layer'],
                            'selected': selected, 'response_sha256': object_hash(row), 'sha256': digest(temporary)})
                        hidden = read_hidden(temporary, row, entry, selected)
                    with torch.inference_mode():
                        feature, metrics = aggregate(hidden, selected, weights, entry, cfg['chunk_size'])
                    quality.append({**detail, 'status': 'ok', 'hidden_sha256': digest(temporary), **metrics})
                    if row['id'] in label_ids:
                        quality[-1]['offsets'] = offsets
                        kept.append(row['id'])
                        features.append(feature)
                    temporaries.append(temporary)
                space(cfg, len(features) * entry['d_sae'] * 20 + 2**20)
                arrays = {k: np.stack([f[k] for f in features]) if features else np.empty((0, d), dtype='int32' if k == 'peaks' else 'float32')
                    for k, d in [('sae_mean', entry['d_sae']), ('sae_max', entry['d_sae']),
                        ('sae_frequency', entry['d_sae']), ('peaks', entry['d_sae']), ('hidden_mean', entry['d_in'])]}
                atomic_npz(base.with_suffix('.npz'), ids=np.asarray(kept, dtype='U40'), **arrays)
                write_rows(base.with_suffix('.jsonl'), quality)
                save(base.with_suffix('.meta.json'), {'signature': bundle_signature,
                    'quality_sha256': digest(base.with_suffix('.jsonl')), 'feature_sha256': digest(base.with_suffix('.npz')),
                    'feature_ids': kept})
                verify_bundle(base, ids, bundle_signature, entry['d_sae'], entry['d_in'], kept)
                for temporary in temporaries:
                    delete_hidden(temporary, base, ids, bundle_signature, entry, kept)
                print(f'{entry["id"]}: pid={problem["pid"]}, ok={sum(q["status"] == "ok" for q in quality)}', flush=True)
            complete = limit is None or limit >= len(problems)
            state = {'signature': signature, 'entry': entry, 'labels_sha256': label_hash,
                'gpu': gpu,
                'complete': complete, 'processed_problems': len(problems[:limit]),
                'seconds_this_invocation': time.monotonic() - start_time,
                'peak_cuda_bytes': torch.cuda.max_memory_allocated(), 'disk_free_bytes': space(cfg)}
            save(state_path, state)
            current[entry['id']] = str(directory.relative_to(out))
            save(out / 'index.json', current)
            del weights
            torch.cuda.empty_cache()
            # Only this owned, checksum-verified checkpoint is disposable; the base model cache stays.
            if checkpoint.parent.resolve() != (Path(cfg['cache_dir']) / 'sae').resolve() or digest(checkpoint) != entry['weights_sha256']:
                raise ValueError('Checkpoint cleanup identity mismatch')
            checkpoint.unlink()


def load_entry(cfg, entry, require_complete=True, include_arrays=True):
    import numpy as np
    out = Path(cfg['run_dir']) / 'extraction'
    index = read_json(out / 'index.json')
    directory = out / index[entry['id']]
    if out.resolve() not in directory.resolve().parents:
        raise ValueError('Extraction index escaped run directory')
    state = read_json(directory / 'run.json')
    if require_complete and not state['complete']:
        raise ValueError('Full extraction required for analysis')
    problems, _ = prepared(cfg)
    quality, ids, blocks = [], [], []
    for problem in problems[:state['processed_problems']]:
        base = directory / f'{problem["pid"]:05d}'
        records = read_problem_responses(cfg, problem)
        meta = read_json(base.with_suffix('.meta.json'))
        signature = object_hash([state['signature'], digest(response_path(cfg, problem['pid']))])
        quality.extend(verify_bundle(base, [r['id'] for r in records], signature,
            entry['d_sae'], entry['d_in'], meta['feature_ids']))
        if meta['feature_ids'] and include_arrays:
            with np.load(base.with_suffix('.npz'), allow_pickle=False) as data:
                ids.extend(data['ids'].tolist())
                blocks.append({k: data[k].copy() for k in data.files if k != 'ids'})
    arrays = {k: np.concatenate([b[k] for b in blocks]) for k in blocks[0]} if blocks else {}
    return ids, arrays, quality, state
