import json
import os

from .common import ASSET_HASHES, atomic_json, digest, download, object_hash, paths, read_json, read_rows


def join_records(rows, tokens, meta_ids):
    by_id = {row['id']: row for row in tokens}
    ids = [row['id'] for row in rows]
    if len(set(ids)) != len(ids) or len(by_id) != len(tokens):
        raise ValueError('Duplicate response/token ID')
    if set(ids) != set(by_id) or ids != meta_ids:
        raise ValueError('Missing/mismatched response IDs or metadata row order')
    result = []
    for row in rows:
        token = by_id[row['id']]
        if row['split'] != 'train' or not isinstance(row['pid'], int):
            raise ValueError('Unexpected source split or problem ID')
        for name in ['prompt_token_ids', 'output_token_ids']:
            if not token[name] or any(type(t) is not int or t < 0 for t in token[name]):
                raise ValueError(f'Invalid token IDs: {row["id"]}')
        if row['mode'] not in ['greedy', 'sampled'] or row['finish'] not in ['stop', 'length']:
            raise ValueError('Unknown generation mode/finish')
        result.append({k: row[k] for k in ['id', 'pid', 'sample', 'mode', 'finish', 'response']})
        result[-1].update({name: token[name] for name in ['prompt_token_ids', 'output_token_ids']})
    return result


def assign_splits(records, seed, fraction):
    import numpy as np
    from sklearn.model_selection import GroupShuffleSplit
    groups = [r['pid'] for r in records]
    train, confirm = next(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
                          .split(np.zeros(len(records)), groups=groups))
    for indices, label in [(train, 'discovery'), (confirm, 'confirmation')]:
        for i in indices:
            records[i]['partition'] = label
    seen = {}
    for row in records:
        key = (row['pid'], row['response'])
        row['canonical_id'] = seen.setdefault(key, row['id'])
    return records


def run(cfg):
    data, out = paths(cfg)
    data.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    if cfg['release'] != 'qwen-gsm8k-self-errors-v1':
        raise ValueError('This loader is pinned to qwen-gsm8k-self-errors-v1')
    base = f'https://github.com/KyeongwooLee/AI-Endoscope/releases/download/{cfg["release"]}/'
    for name, checksum in ASSET_HASHES.items():
        download(base + name, data / name, checksum)
    rows = read_rows(data / 'errors.jsonl')
    records = join_records(rows, read_rows(data / 'token_ids.jsonl'), read_json(data / 'features_meta.json')['ids'])
    if len(records) != cfg['expected_responses'] or len({r['pid'] for r in records}) != cfg['expected_problems']:
        raise ValueError('Release response/problem count mismatch')
    records = assign_splits(records, cfg['split_seed'], cfg['confirmation_fraction'])
    response_pids = {}
    for r in records:
        response_pids.setdefault(object_hash(r['response']), set()).add(r['pid'])
    manifest = {'config': cfg, 'assets': ASSET_HASHES,
                'cross_problem_identical_response_groups': sum(len(v) > 1 for v in response_pids.values())}
    payload = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records)
    target = out / 'prepared.jsonl'
    temp = out / 'prepared.jsonl.tmp'
    temp.write_text(payload, encoding='utf-8')
    manifest['prepared_sha256'] = digest(temp)
    if (out / 'manifest.json').exists() and read_json(out / 'manifest.json') != manifest:
        raise ValueError('Existing run differs; choose a new output directory')
    os.replace(temp, target)
    atomic_json(out / 'manifest.json', manifest)
    atomic_json(out / 'splits.json', {r['id']: r['partition'] for r in records})
    print(f'Prepared {len(records)} frozen completions; no inference performed.')
