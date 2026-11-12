"""Semantic evaluation from independent human reviews. Never fills missing reviews."""
import collections

import numpy as np

from .io import ProtocolError
from .review import read_csv


def load_adjudications(path, pair_codes):
    if path is None or not path.exists():
        return {}
    out = {}
    for row in read_csv(path):
        code = row['pair_code'].lstrip("'")
        final = (row.get('final_judgment') or '').strip().lower()
        if not code and not final:
            continue
        if code not in pair_codes or final not in ('same', 'different', 'uncertain'):
            raise ProtocolError(f'Invalid adjudication row {row}')
        if not (row.get('adjudicator') or '').strip():
            raise ProtocolError(f'Adjudication for {code} lacks adjudicator')
        out[code] = final
    return out


def resolve_pairs(human, adjudications, pair_codes):
    """Final pair judgment. Unreviewed and unresolved disagreements are missing (None)."""
    resolved = {}
    for code in pair_codes:
        votes = [r['pairs'][code]['judgment'] for r in human.values()
                 if code in r['pairs'] and r['pairs'][code]['judgment']]
        if code in adjudications:
            resolved[code] = {'final': adjudications[code], 'state': 'adjudicated', 'n_reviewers': len(votes)}
        elif not votes:
            resolved[code] = {'final': None, 'state': 'unreviewed', 'n_reviewers': 0}
        elif len(set(votes)) == 1:
            resolved[code] = {'final': votes[0], 'state': 'single' if len(votes) == 1 else 'agreed', 'n_reviewers': len(votes)}
        else:
            resolved[code] = {'final': None, 'state': 'disagreement_unadjudicated', 'n_reviewers': len(votes)}
    return resolved


def verified_yield(retrievals, code_of, resolved, k, methods):
    """Per-method per-query verified SAME@k; None for a query with any missing required judgment."""
    by_method = collections.defaultdict(list)
    for res in sorted(retrievals, key=lambda r: (r['method'], r['query_index'])):
        by_method[res['method']].append(res)
    out = {}
    for m in methods:
        rows = []
        for res in by_method[m]:
            judgments = [resolved[code_of[(res['query_id'], s['id'])]]['final'] for s in res['top_k'][:k]]
            missing = sum(j is None for j in judgments)
            same = sum(j == 'same' for j in judgments)
            unc = sum(j == 'uncertain' for j in judgments)
            rows.append({'query_id': res['query_id'], 'query_index': res['query_index'], 'status': res['status'],
                         'returned': len(judgments), 'same': same, 'uncertain': unc, 'missing_judgments': missing,
                         'conservative': None if missing else same / k,
                         'optimistic': None if missing else (same + unc) / k})
        complete = all(r['conservative'] is not None for r in rows)
        out[m] = {'per_query': rows, 'complete': complete,
                  'queries': len(rows), 'queries_with_missing_judgments': sum(r['conservative'] is None for r in rows),
                  'verified_same_error_at_k': float(np.mean([r['conservative'] for r in rows])) if complete and rows else None,
                  'optimistic_bound': float(np.mean([r['optimistic'] for r in rows])) if complete and rows else None,
                  'query_coverage': sum(r['status'] == 'ranked' for r in rows) / len(rows) if rows else None,
                  'conditional_yield_among_ranked_queries':
                      float(np.mean([r['conservative'] for r in rows if r['status'] == 'ranked']))
                      if complete and any(r['status'] == 'ranked' for r in rows) else None}
    return out


def paired_bootstrap(a, b, draws, seed, level):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ProtocolError('Paired bootstrap needs equal-length 1-D arrays')
    idx = np.random.default_rng(seed).integers(0, len(a), size=(draws, len(a)))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    alpha = (1 - level) / 2
    return {'difference': float(a.mean() - b.mean()), 'ci_low': float(np.quantile(diffs, alpha)),
            'ci_high': float(np.quantile(diffs, 1 - alpha)), 'draws': draws, 'level': level,
            'resampling_unit': 'query question (same draw for both methods)'}


def contrasts(yields, cfg):
    b = cfg['bootstrap']
    out = {}
    for x, y in b['contrasts']:
        name = f'{x}-{y}'
        if not (yields[x]['complete'] and yields[y]['complete']):
            out[name] = {'status': 'not_computed_missing_reviews'}
            continue
        ax = [r['conservative'] for r in yields[x]['per_query']]
        ay = [r['conservative'] for r in yields[y]['per_query']]
        out[name] = paired_bootstrap(ax, ay, b['draws'], b['seed'], b['level'])
        ox = [r['optimistic'] for r in yields[x]['per_query']]
        oy = [r['optimistic'] for r in yields[y]['per_query']]
        out[name]['optimistic_difference'] = float(np.mean(ox) - np.mean(oy))
    return out


def description_metrics(human, item_key):
    """Per human reviewer: rates of stage-2 ratings by (unblinded via private key) condition."""
    cond_of = {(r['item_code'], r['description_label']): r['condition'] for r in item_key}
    per_reviewer = {}
    for rid, r in human.items():
        stats = collections.defaultdict(collections.Counter)
        for (code, label), v in r['items_stage2'].items():
            cond = cond_of.get((code, label))
            for field in ('correctness', 'specificity', 'unsupported_assertion', 'evidence_fidelity'):
                if v[field]:
                    stats[cond][f'{field}={v[field]}'] += 1
            stats[cond]['rated'] += bool(v['correctness'])
        stage1 = collections.Counter(v['observable_error_present'] for v in r['items_stage1'].values() if v['observable_error_present'])
        per_reviewer[rid] = {'stage2': {c: dict(s) for c, s in stats.items()}, 'stage1_observable_error_present': dict(stage1)}
    agreement = {}
    ids = sorted(human)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = human[ids[i]]['pairs'], human[ids[j]]['pairs']
            common = [c for c in a if c in b and a[c]['judgment'] and b[c]['judgment']]
            agreement[f'{ids[i]}~{ids[j]}'] = {
                'pairs_both_judged': len(common),
                'raw_agreement': (sum(a[c]['judgment'] == b[c]['judgment'] for c in common) / len(common)) if common else None}
    return {'per_reviewer': per_reviewer, 'pair_agreement': agreement}
