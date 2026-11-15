import json
import random
from pathlib import Path

from . import batch, taxonomy
from .common import (digest, lock, object_hash, prepared, read_json, read_problem_responses,
                     rows, save, write_rows)


def schema(codebook):
    return batch.obj({'status': {'type': 'string', 'enum': ['classified', 'unclassified']},
        'types': {'type': 'array', 'items': {'type': 'string', 'enum': [c['id'] for c in codebook['categories']]}},
        'process_error': {'type': ['boolean', 'null']}, 'first_error_char': {'type': ['integer', 'null']},
        'first_error_step': {'type': ['integer', 'null']},
        'evidence': {'type': 'array', 'items': batch.obj({'type': batch.STR, 'start': {'type': 'integer'},
            'end': {'type': 'integer'}, 'quote': batch.STR, 'reason': batch.STR})},
        'unclassified_reason': batch.STR})


def validate(value, row, codebook):
    if not isinstance(value, dict) or set(value) != set(schema(codebook)['properties']):
        raise ValueError('Label object has missing/unknown fields')
    allowed = {c['id'] for c in codebook['categories']}
    types = value['types']
    if len(set(types)) != len(types) or not set(types) <= allowed:
        raise ValueError('Invalid/duplicate type labels')
    if value['status'] == 'unclassified':
        if (not value['unclassified_reason'].strip() or types or value['process_error'] is not None
                or value['evidence'] or value['first_error_char'] is not None or value['first_error_step'] is not None):
            raise ValueError('Unclassified must retain uncertainty without forced types')
        return value
    if value['status'] != 'classified' or type(value['process_error']) is not bool:
        raise ValueError('Invalid classified label')
    if value['process_error'] != bool(types) or bool(value['unclassified_reason']):
        raise ValueError('Process-error/types mismatch')
    evidence = value['evidence']
    for item in evidence:
        a, b = item['start'], item['end']
        if (type(a) is not int or type(b) is not int or not 0 <= a < b <= len(row['response'])
                or row['response'][a:b] != item['quote'] or item['type'] not in types or not item['reason'].strip()):
            raise ValueError('Evidence does not match response character offsets/type')
    if types:
        if (set(e['type'] for e in evidence) != set(types)
                or type(value['first_error_char']) is not int
                or value['first_error_char'] != min(e['start'] for e in evidence)
                or type(value['first_error_step']) is not int or value['first_error_step'] < 1):
            raise ValueError('Missing per-type evidence or earliest error')
    elif evidence or value['first_error_char'] is not None or value['first_error_step'] is not None:
        raise ValueError('Error-free response has error evidence')
    return value


def selection(cfg, scope, count):
    problems, _ = prepared(cfg)
    if scope == 'discovery':
        return taxonomy.sample(cfg)
    choices = [p for p in problems if p['partition'] == 'confirmation']
    random.Random(cfg['seed']).shuffle(choices)
    if scope == 'pilot':
        return choices[:cfg['pilot_problems']]
    if scope != 'confirmation' or count is not None and count < 1:
        raise ValueError('Confirmation needs a positive --problems count')
    review_pilot(cfg)
    return choices[cfg['pilot_problems']:None if count is None else cfg['pilot_problems'] + count]


def build(cfg, scope, count=None):
    code = taxonomy.frozen(cfg)
    with lock(Path(cfg['run_dir']) / 'labels'):
        selected = selection(cfg, scope, count)
        _, book = batch.ledger(cfg)
        existing = {rid for b in book['batches'].values() if b['kind'] == 'labels'
                    for rid in b['context']['mapping']}
        # Select complete problem groups; never cherry-pick completions to fit the budget.
        mapping, requests, pending, funded = {}, [], [], []
        remaining = batch.micros(cfg['labeling']['budget_usd']) - batch.charged(book)
        for problem in selected:
            group = read_problem_responses(cfg, problem)
            if all(r['id'] in existing for r in group):
                continue
            if any(r['id'] in existing for r in group):
                raise ValueError('Partial prior problem submission detected; recover its batch first')
            canonical, group_mapping, group_requests = {}, {}, []
            for row in group:
                canonical_id = canonical.setdefault(row['response'], row['id'])
                group_mapping[row['id']] = canonical_id
                if canonical_id != row['id']:
                    continue
                group_requests.append(batch.request(row['id'],
                    'Inspect the entire reasoning using the frozen codebook, including correct final answers. '
                    'Return multiple observable types, first erroneous reasoning step (1-based), and exact evidence quotes '
                    'with zero-based Python character offsets [start,end). Earliest error is the minimum evidence start. '
                    'Use classified/types=[]/process_error=false for sound reasoning; use unclassified with reason when '
                    'evidence is insufficient. Do not infer a cognitive cause from the answer.',
                    {'codebook': code['codebook'], 'question': row['question'], 'gold': row['gold'],
                     'response': row['response'], 'id': row['id']}, schema(code['codebook']), cfg))
            price = sum(batch.upper_cost(r, cfg) for r in group_requests)
            if price > remaining:
                if scope in ('pilot', 'discovery'):
                    raise ValueError('Budget cannot cover the complete discovery/pilot group; nothing submitted')
                break
            funded.append(problem['pid'])
            remaining -= price
            pending.extend(group)
            requests.extend(group_requests)
            mapping.update(group_mapping)
        if not pending:
            prior = [key for key,b in book['batches'].items() if b['kind'] == 'labels'
                     and set(b['context']['pids']) & {p['pid'] for p in selected}]
            if prior:
                print('No new funded groups; resume existing keys: ' + ', '.join(prior))
                return prior
            raise ValueError('Budget cannot cover one complete problem; nothing submitted')
        print(f'Selected {len(funded)} whole problems / {len(pending)} responses under the shared budget')
        return batch.plan(cfg, 'labels', requests, {'scope': scope, 'mapping': mapping,
            'pids': list(dict.fromkeys(r['pid'] for r in pending)), 'codebook_sha256': code['sha256'],
            'responses_sha256': object_hash(pending)})


def materialize(cfg):
    code = taxonomy.frozen(cfg)
    _, book = batch.ledger(cfg)
    problems, _ = prepared(cfg)
    by_pid = {p['pid']: p for p in problems}
    output = {}
    for key, value in book['batches'].items():
        if value['kind'] != 'labels' or value['state'] != 'collected':
            continue
        context = value['context']
        if context['codebook_sha256'] != code['sha256']:
            raise ValueError('Label codebook differs from frozen taxonomy')
        original = [r for pid in context['pids'] for r in read_problem_responses(cfg, by_pid[pid])]
        if object_hash(original) != context['responses_sha256']:
            raise ValueError('Responses changed after label request creation')
        result = {r['custom_id']: r for r in batch.results(cfg, key)}
        for row in original:
            canonical = context['mapping'][row['id']]
            try:
                label = validate(batch.parse_result(result[canonical]), row, code['codebook'])
            except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
                label = {'status': 'unclassified', 'types': [], 'process_error': None,
                    'first_error_char': None, 'first_error_step': None, 'evidence': [],
                    'unclassified_reason': f'API_or_validation_failure: {exc}'}
            output[row['id']] = {'id': row['id'], 'pid': row['pid'], 'partition': row['partition'],
                'canonical_id': canonical, 'scope': context['scope'], 'batch_key': key,
                'codebook_sha256': code['sha256'], 'label': label}
    path = Path(cfg['run_dir']) / 'labels' / 'labels.jsonl'
    write_rows(path, [output[k] for k in sorted(output)])
    save(path.with_suffix('.meta.json'), {'sha256': digest(path), 'codebook_sha256': code['sha256']})
    return path


def load(cfg):
    path = Path(cfg['run_dir']) / 'labels' / 'labels.jsonl'
    meta = read_json(path.with_suffix('.meta.json'))
    if digest(path) != meta['sha256'] or meta['codebook_sha256'] != taxonomy.frozen(cfg)['sha256']:
        raise ValueError('Label file corrupted or codebook changed')
    values = list(rows(path))
    if len({v['id'] for v in values}) != len(values):
        raise ValueError('Duplicate materialized labels')
    return values


def review_pilot(cfg, review_file=None):
    if review_file:
        data = read_json(review_file)
        target = Path(cfg['run_dir']) / 'labels' / 'pilot.review.json'
    else:
        target = Path(cfg['run_dir']) / 'labels' / 'pilot.review.json'
        data = read_json(target)
    pilot = [r for r in load(cfg) if r['scope'] == 'pilot']
    if (not pilot or len({r['pid'] for r in pilot}) != cfg['pilot_problems']
            or len(pilot) != cfg['pilot_problems'] * cfg['repeats']):
        raise ValueError('Complete problem-level label pilot required')
    _, book = batch.ledger(cfg)
    if any(not book['batches'][r['batch_key']].get('usage_complete') for r in pilot):
        raise ValueError('Pilot actual API usage is not fully reconciled')
    if (data.get('pilot_sha256') != object_hash(pilot) or not data.get('reviewed_by', '').strip()
            or data.get('approved') is not True or not data.get('evidence_quality_notes', '').strip()
            or not data.get('actual_cost_notes', '').strip()):
        raise ValueError(f'Pilot review must include approved=true, reviewer, evidence/cost notes and pilot_sha256={object_hash(pilot)}')
    if review_file:
        save(target, data)
    return data
