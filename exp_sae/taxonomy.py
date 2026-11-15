import random
from pathlib import Path

from . import batch
from .common import (digest, object_hash, prepared, read_json, read_problem_responses,
                     save, write_rows)


CATEGORY = batch.obj({'id': batch.STR, 'name': batch.STR, 'definition': batch.STR,
    'include': batch.STR, 'exclude': batch.STR,
    'examples': {'type': 'array', 'items': batch.obj({'response_id': batch.STR,
        'quote': batch.STR, 'explanation': batch.STR})}})
SCHEMA = batch.obj({'categories': {'type': 'array', 'items': CATEGORY},
    'multi_label_rules': batch.STR, 'first_error_rules': batch.STR, 'unclassified_rules': batch.STR})


def sample(cfg):
    problems, _ = prepared(cfg)
    choices = [p for p in problems if p['partition'] == 'discovery']
    if len(choices) < cfg['discovery_problems']:
        raise ValueError('Too few discovery problems')
    return random.Random(cfg['seed']).sample(choices, cfg['discovery_problems'])


def build(cfg, synthesize=False):
    selected = sample(cfg)
    records = [r for p in selected for r in read_problem_responses(cfg, p)]
    out = Path(cfg['run_dir']) / 'taxonomy'
    write_rows(out / 'discovery_400.jsonl', records)
    context = {'pids': [p['pid'] for p in selected], 'responses_sha256': digest(out / 'discovery_400.jsonl')}
    if not synthesize:
        requests = []
        for problem in selected:
            payload = {'question': problem['question'], 'gold': problem['gold'],
                'responses': [{'id': r['id'], 'response': r['response']} for r in records if r['pid'] == problem['pid']]}
            requests.append(batch.request(str(problem['pid']),
                'Develop candidate error categories from all 20 responses. Check reasoning even when final answer is correct. '
                'Use observable mathematical errors, explicit boundaries, exact evidence quotes and response IDs. '
                'Allow multiple types, earliest error and insufficient evidence. Do not force a legacy taxonomy.', payload, SCHEMA, cfg))
        return batch.plan(cfg, 'taxonomy-discovery', requests, context)
    _, book = batch.ledger(cfg)
    matches = [key for key, value in book['batches'].items()
               if value['kind'] == 'taxonomy-discovery' and value['context'] == context]
    if len(matches) != 1:
        raise ValueError('Build/collect the per-problem discovery batch first')
    drafts = [batch.parse_result(line) for line in batch.results(cfg, matches[0])]
    if len(drafts) != len(selected):
        raise ValueError('All discovery problems must have successful drafts before synthesis')
    req = batch.request('codebook',
        'Merge these exploratory per-problem taxonomies into one concise codebook with definitions, inclusion/exclusion '
        'boundaries, verbatim observed examples, multi-label, earliest-error and unclassified rules. '
        'Preserve evidence response IDs. Do not assert that categories are validated causes.', drafts, SCHEMA, cfg)
    context['draft_batch'] = matches[0]
    return batch.plan(cfg, 'taxonomy-synthesis', [req], context)


def export_draft(cfg, key):
    _, book = batch.ledger(cfg)
    if book['batches'][key]['kind'] != 'taxonomy-synthesis':
        raise ValueError('Expected taxonomy synthesis batch')
    result = batch.results(cfg, key)
    if len(result) != 1:
        raise ValueError('Synthesis result missing')
    draft = {'version': 'v1', 'reviewed_by': '', 'review_notes': '', **batch.parse_result(result[0]),
             'source_batch': key}
    path = Path(cfg['run_dir']) / 'taxonomy' / 'codebook.draft.json'
    if not path.exists():
        save(path, draft)
    return path


def freeze(cfg, draft_path):
    value = read_json(draft_path)
    if not value.get('reviewed_by', '').strip() or not value.get('review_notes', '').strip() or not value.get('version'):
        raise ValueError('Human codebook review, notes and version are required')
    categories = value['categories']
    ids = [c['id'] for c in categories]
    if not ids or len(set(ids)) != len(ids) or any(not i or i in ('correct', 'unclassified') for i in ids):
        raise ValueError('Invalid category IDs')
    records = {r['id']: r for p in sample(cfg) for r in read_problem_responses(cfg, p)}
    for category in categories:
        if any(not category.get(k, '').strip() for k in ('name', 'definition', 'include', 'exclude')) or not category['examples']:
            raise ValueError('Each category needs boundaries and observed examples')
        for example in category['examples']:
            if (example['response_id'] not in records or not example['quote']
                    or example['quote'] not in records[example['response_id']]['response']):
                raise ValueError('Codebook example is not an exact discovery response quote')
    for key in ('multi_label_rules', 'first_error_rules', 'unclassified_rules'):
        if not value[key].strip():
            raise ValueError(f'Missing {key}')
    frozen = {'codebook': value, 'sha256': object_hash(value),
              'discovery_response_hash': object_hash(records)}
    target = Path(cfg['run_dir']) / 'taxonomy' / 'codebook.frozen.json'
    if target.exists() and read_json(target) != frozen:
        raise ValueError('Codebook already frozen; use a new run for revisions')
    save(target, frozen)


def frozen(cfg):
    value = read_json(Path(cfg['run_dir']) / 'taxonomy' / 'codebook.frozen.json')
    if object_hash(value['codebook']) != value['sha256']:
        raise ValueError('Frozen codebook hash mismatch')
    records = {r['id']: r for p in sample(cfg) for r in read_problem_responses(cfg, p)}
    if value['discovery_response_hash'] != object_hash(records):
        raise ValueError('Discovery responses changed after codebook review')
    return value
