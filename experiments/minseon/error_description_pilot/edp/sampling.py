"""Shared SAE split reproduction, eligibility ledger and bounded pilot sampling."""
import collections

import numpy as np

from .io import ProtocolError

STREAMS = {'dev_uniform_groups': 1, 'dev_uniform_response': 2, 'dev_enriched': 3,
           'eval_groups': 4, 'eval_response': 5, 'eval_roles': 6}


def assign_sae_partitions(pids, random_state=42, test_size=0.2):
    """Exactly Kyeongwoo's prepare.assign_splits on the full ordered rows (before filtering)."""
    from sklearn.model_selection import GroupShuffleSplit
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train, confirm = next(splitter.split(np.zeros(len(pids)), groups=list(pids)))
    labels = [None] * len(pids)
    for i in train:
        labels[i] = 'discovery'
    for i in confirm:
        labels[i] = 'confirmation'
    if any(label is None for label in labels):
        raise ProtocolError('Split left unassigned rows')
    return labels


def split_summary(records):
    out = {}
    for part in ('discovery', 'confirmation'):
        rows = [r for r in records if r['sae_partition'] == part]
        out[f'{part}_responses'] = len(rows)
        out[f'{part}_questions'] = len({r['source_pid'] for r in rows})
    return out


def rng_for(seed, stream):
    return np.random.default_rng([seed, STREAMS[stream]])


def build_eligibility(rows, records, prior_ledger):
    """Return (units, exclusions, counts).

    A unit is one canonical exact (pid, response) pair with every source ID mapped to it.
    """
    copies = collections.defaultdict(list)
    for row, rec in zip(rows, records):
        copies[rec['canonical_response_id']].append(rec['source_id'])
    exclusions = []
    by_id = {rec['source_id']: (row, rec) for row, rec in zip(rows, records)}
    for row, rec in zip(rows, records):
        if rec['canonical_response_id'] != rec['source_id']:
            exclusions.append({'level': 'response', 'source_id': rec['source_id'],
                               'reason': 'exact_duplicate_same_question_response',
                               'canonical_response_id': rec['canonical_response_id']})
    # Cross-partition normalized-question conflicts -> quarantine (never re-split).
    parts = collections.defaultdict(set)
    for rec in records:
        parts[rec['question_sha256']].add(rec['sae_partition'])
    conflicts = {q for q, p in parts.items() if len(p) > 1}
    units = []
    for canonical, ids in copies.items():
        row, rec = by_id[canonical]
        finishes = {by_id[i][1]['source_finish'] for i in ids}
        if len(finishes) != 1:
            raise ProtocolError(f'Identical responses with different finish: {canonical}')
        reasons = []
        if rec['source_finish'] == 'length':
            reasons.append('truncated_finish_length')
        if rec['question_sha256'] in conflicts:
            reasons.append('cross_partition_question_conflict')
        reasons.extend(prior_ledger.get(rec['question_key'], []))
        unit = {**rec, 'all_source_ids': sorted(ids, key=lambda s: by_id[s][1]['source_row_index']),
                'response_chars': len(row['response']), 'rule_type_hidden': row['rule_type'],
                'eligible': not reasons, 'exclusion_reasons': reasons}
        units.append(unit)
        if 'truncated_finish_length' in reasons:
            exclusions.append({'level': 'response', 'source_id': canonical, 'reason': 'truncated_finish_length',
                               'duplicate_source_ids': ids})
        if 'cross_partition_question_conflict' in reasons:
            exclusions.append({'level': 'question', 'question_key': rec['question_key'],
                               'reason': 'cross_partition_question_conflict'})
    for qkey, reasons in sorted(prior_ledger.items()):
        for reason in reasons:
            exclusions.append({'level': 'question', 'question_key': qkey, 'reason': reason})

    def qcount(pred, part):
        return len({u['source_pid'] for u in units if u['sae_partition'] == part and pred(u)})
    labels_only = lambda u: (u['source_finish'] == 'stop' and not any(
        r.startswith('prior_annotation_question') for r in u['exclusion_reasons']))
    counts = {
        'canonical_units': len(units),
        'cross_partition_question_conflicts': len(conflicts),
        'questions_after_length_and_labels_jsonl_exclusion': {
            'discovery': qcount(labels_only, 'discovery'), 'confirmation': qcount(labels_only, 'confirmation')},
        'questions_after_all_exclusions': {
            'discovery': qcount(lambda u: u['eligible'], 'discovery'),
            'confirmation': qcount(lambda u: u['eligible'], 'confirmation')},
        'eligible_units_after_all_exclusions': {
            'discovery': sum(u['eligible'] and u['sae_partition'] == 'discovery' for u in units),
            'confirmation': sum(u['eligible'] and u['sae_partition'] == 'confirmation' for u in units)},
    }
    return units, exclusions, counts


def _groups(units, partition):
    groups = collections.defaultdict(list)
    for u in units:
        if u['eligible'] and u['sae_partition'] == partition:
            groups[u['source_pid']].append(u)
    for pid in groups:
        groups[pid].sort(key=lambda u: u['source_row_index'])
    return dict(sorted(groups.items()))


def _stratum_match(unit, stratum, short_max):
    if stratum == 'SHORT_RESPONSE':
        return unit['response_chars'] <= short_max
    return unit['rule_type_hidden'] == stratum


def sample_pilot(units, cfg):
    """Question groups first, then one response within each selected group."""
    s = cfg['sampling']
    seed = s['seed']
    dev_groups = _groups(units, 'discovery')
    eval_groups = _groups(units, 'confirmation')
    selected = []

    keys = list(dev_groups)
    rng = rng_for(seed, 'dev_uniform_groups')
    chosen = sorted(int(keys[i]) for i in rng.choice(len(keys), s['dev_uniform'], replace=False))
    rng_r = rng_for(seed, 'dev_uniform_response')
    for pid in chosen:
        options = dev_groups[pid]
        selected.append((options[int(rng_r.integers(len(options)))], 'development', 'uniform'))
    used = set(chosen)
    rng_e = rng_for(seed, 'dev_enriched')
    enrichment = []
    for stratum, n in s['dev_enriched_strata']:
        cands = [pid for pid, us in dev_groups.items() if pid not in used
                 and any(_stratum_match(u, stratum, s['short_response_max_chars']) for u in us)]
        take = min(n, len(cands))
        picks = sorted(int(cands[i]) for i in rng_e.choice(len(cands), take, replace=False)) if take else []
        for pid in picks:
            options = [u for u in dev_groups[pid] if _stratum_match(u, stratum, s['short_response_max_chars'])]
            selected.append((options[int(rng_e.integers(len(options)))], 'development', f'enriched:{stratum}'))
            used.add(pid)
        enrichment.append({'stratum': stratum, 'requested': n, 'candidates': len(cands), 'selected': take})

    keys = list(eval_groups)
    if len(keys) < s['eval_questions']:
        raise ProtocolError('Too few eligible confirmation questions')
    rng = rng_for(seed, 'eval_groups')
    chosen = sorted(int(keys[i]) for i in rng.choice(len(keys), s['eval_questions'], replace=False))
    rng_r = rng_for(seed, 'eval_response')
    eval_units = [eval_groups[pid][int(rng_r.integers(len(eval_groups[pid])))] for pid in chosen]
    order = rng_for(seed, 'eval_roles').permutation(len(eval_units))
    query_idx = set(int(i) for i in order[:s['eval_queries']])
    for i, unit in enumerate(eval_units):
        role = 'eval_query' if i in query_idx else 'eval_gallery'
        selected.append((unit, role, 'uniform'))

    manifest = []
    for unit, role, how in selected:
        manifest.append({k: unit[k] for k in (
            'source_release', 'source_asset_sha256', 'source_row_index', 'source_id', 'namespaced_id',
            'source_pid', 'question_key', 'question_sha256', 'response_sha256', 'canonical_response_id',
            'all_source_ids', 'source_mode', 'source_finish', 'sae_partition')}
            | {'pilot_role': role, 'selection': how})
    manifest.sort(key=lambda m: ({'development': 0, 'eval_query': 1, 'eval_gallery': 2}[m['pilot_role']], m['source_pid']))
    check_sample(manifest, cfg)
    return manifest, {'enrichment': enrichment, 'eligible_dev_groups': len(dev_groups),
                      'eligible_eval_groups': len(eval_groups),
                      'rng': f'numpy.random.default_rng([seed, stream]) numpy=={np.__version__}',
                      'streams': STREAMS, 'seed': seed}


def check_sample(manifest, cfg):
    s = cfg['sampling']
    roles = collections.Counter(m['pilot_role'] for m in manifest)
    expected_dev = s['dev_uniform'] + sum(n for _, n in s['dev_enriched_strata'])
    if roles['eval_query'] != s['eval_queries'] or roles['eval_gallery'] != s['eval_questions'] - s['eval_queries']:
        raise ProtocolError(f'Unexpected eval role counts {roles}')
    if roles['development'] > expected_dev:
        raise ProtocolError('Too many development rows')
    for key in ('source_pid', 'question_sha256', 'response_sha256'):
        values = [m[key] for m in manifest]
        if len(set(values)) != len(values):
            raise ProtocolError(f'Duplicate {key} across pilot rows (dev/query/gallery must be distinct)')
    for m in manifest:
        want = 'discovery' if m['pilot_role'] == 'development' else 'confirmation'
        if m['sae_partition'] != want:
            raise ProtocolError(f'{m["source_id"]} role {m["pilot_role"]} in partition {m["sae_partition"]}')
