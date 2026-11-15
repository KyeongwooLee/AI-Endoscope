"""Curated public metadata; only verified bytes are loaded, never repository code."""
import urllib.request
from pathlib import Path

from .common import digest, object_hash, read_json, save, space


def registry(cfg):
    value = read_json(Path(__file__).with_name('sae_registry.json'))
    for entry in value['entries']:
        if entry['status'] == 'metadata_verified':
            if entry['model'] != cfg['model'] or entry['d_in'] != 1536 or not 0 <= entry['layer'] < 28:
                raise ValueError('Registry model/dimension/layer mismatch')
            if entry['model_revision'] not in (None, cfg['model_revision']):
                raise ValueError('Registry backbone revision mismatch')
    target = Path(cfg['run_dir']) / 'sae_registry.json'
    if target.exists() and read_json(target) != value:
        raise ValueError('Registry changed during run; preserve old artifacts')
    save(target, value)
    return value


def download_weights(cfg, entry):
    path = Path(cfg['cache_dir']) / 'sae' / (entry['weights_sha256'] + Path(entry['weights_file']).suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if digest(path) != entry['weights_sha256']:
            raise ValueError('Cached SAE checkpoint hash mismatch; preserve and inspect')
        return path
    space(cfg, entry['weights_bytes'] * 2)
    url = f'https://huggingface.co/{entry["repo"]}/resolve/{entry["revision"]}/{entry["weights_file"]}'
    request = urllib.request.Request(url, headers={'User-Agent': 'exp-sae/1'})
    temp = path.with_suffix(path.suffix + '.download')
    with urllib.request.urlopen(request, timeout=120) as response, temp.open('wb') as handle:
        while data := response.read(2**20):
            space(cfg, len(data))
            handle.write(data)
    if temp.stat().st_size != entry['weights_bytes'] or digest(temp) != entry['weights_sha256']:
        raise ValueError('Downloaded SAE checksum/size mismatch; partial file retained')
    temp.replace(path)
    return path


def normalize_weights(weights, entry):
    import torch
    if entry['format'] == 'brainpatch':
        config = weights['config']
        if any(config[k] != entry[k] for k in ('d_in', 'd_sae', 'k', 'input_scale', 'model_revision', 'layer')):
            raise ValueError('BrainPatch checkpoint config differs from pinned metadata')
        weights = weights['state_dict']
        result = {k: weights[k] for k in ('W_enc', 'W_dec', 'b_enc', 'b_dec')}
        result['W_enc'] = result['W_enc'].T
        result['W_dec'] = result['W_dec'].T
    elif entry['format'] == 'eleuther':
        if set(weights) != {'encoder.weight', 'encoder.bias', 'W_dec', 'b_dec'}:
            raise ValueError('Unknown Eleuther checkpoint tensors')
        result = {'W_enc': weights['encoder.weight'].T, 'b_enc': weights['encoder.bias'],
                  'W_dec': weights['W_dec'], 'b_dec': weights['b_dec']}
    else:
        result = weights
        expected_keys = {'W_enc', 'W_dec', 'b_enc', 'b_dec'}
        if entry['architecture'] == 'jumprelu':
            expected_keys.add('threshold')
        if set(result) != expected_keys:
            raise ValueError('Unknown SAE-Lens tensors')
    d, width = entry['d_in'], entry['d_sae']
    shapes = {'W_enc': (d, width), 'W_dec': (width, d), 'b_enc': (width,), 'b_dec': (d,)}
    if entry['architecture'] == 'jumprelu':
        shapes['threshold'] = (width,)
    for key, shape in shapes.items():
        if tuple(result[key].shape) != shape or not torch.isfinite(result[key]).all():
            raise ValueError(f'Invalid SAE tensor {key}')
    return {key: result[key].float().to('cuda:0') for key in shapes}


def load(cfg, entry):
    path = download_weights(cfg, entry)
    if entry['format'] == 'brainpatch':
        import torch
        weights = torch.load(path, map_location='cpu', weights_only=True)
    else:
        from safetensors.torch import load_file
        weights = load_file(path, device='cpu')
    return normalize_weights(weights, entry), path


def encode(hidden, weights, entry):
    import torch
    h = hidden.float() * entry['input_scale']
    if entry['apply_b_dec_to_input']:
        h = h - weights['b_dec']
    pre = h @ weights['W_enc'] + weights['b_enc']
    if entry['architecture'] == 'jumprelu':
        z = pre.relu() * (pre > weights['threshold'])
    elif entry['architecture'] == 'topk':
        values, indices = pre.relu().topk(entry['k'], dim=-1)
        z = torch.zeros_like(pre).scatter_(-1, indices, values)
    elif entry['architecture'] == 'standard':
        z = pre.relu()
    else:
        raise ValueError('Unsupported SAE architecture')
    reconstruction = (z @ weights['W_dec'] + weights['b_dec']) / entry['input_scale']
    return z, reconstruction
