"""Blinded human-review packages: description items and pooled query-neighbor pairs."""
import csv
import html
import io
from pathlib import Path

import numpy as np

from .io import ProtocolError, sha256_text, write_json, write_jsonl, write_text

PAIR_JUDGMENTS = ('same', 'different', 'uncertain')
YES_NO_UNSURE = ('yes', 'no', 'unsure')
ITEM1_FIELDS = ['item_code', 'question', 'model_solution', 'reference_solution',
                'observable_error_present', 'observed_error_or_uncertainty', 'reference_problem', 'reviewer_notes']
ITEM2_FIELDS = ['item_code', 'description_label', 'description_text',
                'correctness', 'specificity', 'unsupported_assertion', 'evidence_fidelity', 'reviewer_notes']
PAIR_FIELDS = ['pair_code', 'response_1_question', 'response_1_solution', 'response_1_reference',
               'response_2_question', 'response_2_solution', 'response_2_reference',
               'judgment', 'same_topic', 'same_operation', 'same_style', 'reviewer_notes']
ITEM2_VALUES = {'correctness': ('correct', 'partially_correct', 'incorrect', 'not_applicable'),
                'specificity': ('specific', 'generic', 'not_applicable'),
                'unsupported_assertion': ('yes', 'no', 'not_applicable'),
                'evidence_fidelity': ('faithful', 'unfaithful', 'not_applicable')}
ITEM1_VALUES = {'observable_error_present': ('yes', 'no', 'uncertain'), 'reference_problem': YES_NO_UNSURE}
PAIR_VALUES = {'judgment': PAIR_JUDGMENTS, 'same_topic': YES_NO_UNSURE, 'same_operation': YES_NO_UNSURE,
               'same_style': YES_NO_UNSURE}
ABSTAIN_TEXT = '(no description: the annotator abstained or produced no valid description)'


def spreadsheet_safe(value):
    """Prefix cells that a spreadsheet would interpret as formulas (original text kept in JSONL)."""
    text = '' if value is None else str(value)
    return "'" + text if text[:1] in ('=', '+', '-', '@', '\t', '\r') else text


def csv_text(rows, fields):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction='raise', quoting=csv.QUOTE_ALL, lineterminator='\n')
    writer.writeheader()
    for row in rows:
        writer.writerow({f: spreadsheet_safe(row.get(f, '')) for f in fields})
    return buf.getvalue()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def _codes(prefix, n, rng):
    values = rng.choice(10 ** 6, size=n, replace=False)
    return [f'{prefix}-{int(v):06d}' for v in values]


def html_page(title, blocks):
    css = ('body{font-family:system-ui,sans-serif;max-width:1100px;margin:auto;padding:16px}'
           'pre{white-space:pre-wrap;background:#f5f5f5;padding:8px;border-radius:4px}'
           '.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}section{border-top:2px solid #999;margin-top:24px}')
    return (f'<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>'
            f'<style>{css}</style></head><body><h1>{html.escape(title)}</h1>' + ''.join(blocks) + '</body></html>')


def _pre(label, text):
    return f'<h4>{html.escape(label)}</h4><pre>{html.escape(text)}</pre>'


def write_preserving(path, text, log):
    """Never overwrite a file whose content differs (it may contain reviewer edits)."""
    path = Path(path)
    if path.exists():
        if path.read_text(encoding='utf-8') == text:
            log.append({'file': str(path), 'action': 'unchanged'})
        else:
            log.append({'file': str(path), 'action': 'preserved_existing_differs'})
        return
    write_text(path, text)
    log.append({'file': str(path), 'action': 'written'})


def build_pairs(retrievals, k):
    pairs = {}
    for res in retrievals:
        for slot in res['top_k'][:k]:
            key = (res['query_id'], slot['id'])
            pairs.setdefault(key, []).append({'method': res['method'], 'rank': slot['rank']})
    return pairs


def build_review_package(run_dir, eval_units, rows_by_id, overlays, retrievals, cfg):
    run_dir = Path(run_dir)
    pkg, key_dir = run_dir / 'review_package', run_dir / 'review_key_private'
    rng = np.random.default_rng([cfg['review']['assignment_seed'], 11])
    log = []
    units = sorted(eval_units, key=lambda u: u['source_id'])
    item_codes = _codes('I', len(units), rng)
    order = rng.permutation(len(units))
    stage1, stage2, item_key = [], [], []
    for idx in order:
        u, code = units[idx], item_codes[idx]
        row = rows_by_id[u['source_id']]
        stage1.append({'item_code': code, 'question': row['question'], 'model_solution': row['response'],
                       'reference_solution': row['reference']})
        labels = ['A', 'B']
        conds = ['E', 'G'] if rng.random() < 0.5 else ['G', 'E']
        for label, cond in zip(labels, conds):
            rec = overlays[cond].get(u['source_id'])
            text = rec.get('embedding_text') if rec else None
            stage2.append({'item_code': code, 'description_label': label, 'description_text': text or ABSTAIN_TEXT})
            item_key.append({'item_code': code, 'description_label': label, 'condition': cond,
                             'source_id': u['source_id'], 'pilot_role': u['pilot_role'],
                             'annotation_status': rec['status'] if rec else 'missing'})

    k = cfg['retrieval']['k']
    pairs = build_pairs(retrievals, k)
    keys = sorted(pairs)
    pair_codes = _codes('P', len(keys), rng)
    porder = rng.permutation(len(keys))
    pair_rows, pair_key = [], []
    for idx in porder:
        (qid, gid), code = keys[idx], pair_codes[idx]
        left_query = bool(rng.random() < 0.5)
        a, b = (qid, gid) if left_query else (gid, qid)
        ra, rb = rows_by_id[a], rows_by_id[b]
        pair_rows.append({'pair_code': code, 'response_1_question': ra['question'], 'response_1_solution': ra['response'],
                          'response_1_reference': ra['reference'], 'response_2_question': rb['question'],
                          'response_2_solution': rb['response'], 'response_2_reference': rb['reference']})
        pair_key.append({'pair_code': code, 'query_id': qid, 'gallery_id': gid, 'response_1_is_query': left_query,
                         'returned_by': sorted(pairs[(qid, gid)], key=lambda m: m['method'])})

    # Prespecified second-review subset: all query items + 20% of other items and of pairs.
    frac = cfg['review']['second_review_fraction']
    role = {u['source_id']: u['pilot_role'] for u in units}
    code_by_sid = {units[i]['source_id']: item_codes[i] for i in range(len(units))}
    gallery_codes = sorted(code_by_sid[s] for s in code_by_sid if role[s] == 'eval_gallery')
    query_codes = sorted(code_by_sid[s] for s in code_by_sid if role[s] == 'eval_query')
    n_g = int(round(frac * len(gallery_codes)))
    n_p = int(round(frac * len(pair_rows)))
    second_items = sorted(query_codes + [gallery_codes[i] for i in rng.choice(len(gallery_codes), n_g, replace=False)])
    all_pair_codes = sorted(r['pair_code'] for r in pair_rows)
    second_pairs = sorted(all_pair_codes[i] for i in rng.choice(len(all_pair_codes), n_p, replace=False)) if n_p else []
    assignment = {'reviewer_1': {'items': 'all', 'pairs': 'all'},
                  'reviewer_2': {'items': second_items, 'pairs': second_pairs,
                                 'rule': f'all {len(query_codes)} query items + {n_g} random gallery items '
                                         f'+ {n_p} random pairs ({frac:.0%})'},
                  'seed': cfg['review']['assignment_seed']}

    tpl = pkg / 'templates'
    write_preserving(tpl / 'reviewer_1' / 'items_stage1.csv', csv_text(stage1, ITEM1_FIELDS), log)
    write_preserving(tpl / 'reviewer_1' / 'items_stage2.csv', csv_text(stage2, ITEM2_FIELDS), log)
    write_preserving(tpl / 'reviewer_1' / 'pairs.csv', csv_text(pair_rows, PAIR_FIELDS), log)
    si, sp = set(second_items), set(second_pairs)
    write_preserving(tpl / 'reviewer_2' / 'items_stage1.csv', csv_text([r for r in stage1 if r['item_code'] in si], ITEM1_FIELDS), log)
    write_preserving(tpl / 'reviewer_2' / 'items_stage2.csv', csv_text([r for r in stage2 if r['item_code'] in si], ITEM2_FIELDS), log)
    write_preserving(tpl / 'reviewer_2' / 'pairs.csv', csv_text([r for r in pair_rows if r['pair_code'] in sp], PAIR_FIELDS), log)
    reviewer_json = ('{\n  "reviewer_id": "reviewer_1",\n  "reviewer_type": "human",\n  "initials": "",\n'
                     '  "review_dates": "",\n  "independent_of_other_reviewer": true,\n  "saw_method_key_or_scores": false,\n'
                     '  "notes": ""\n}\n')
    for r in ('reviewer_1', 'reviewer_2'):
        write_preserving(tpl / r / 'reviewer.json', reviewer_json.replace('reviewer_1', r), log)
    write_preserving(pkg / 'adjudication_template.csv',
                     csv_text([], ['pair_code', 'final_judgment', 'adjudicator', 'notes']), log)
    write_jsonl(pkg / 'items_stage1_exact_text.jsonl', stage1)
    write_jsonl(pkg / 'pairs_exact_text.jsonl', pair_rows)
    write_text(pkg / 'items_stage1.html', html_page('Stage 1: diagnose without descriptions', [
        f'<section><h2>{html.escape(r["item_code"])}</h2>' + _pre('Question', r['question']) +
        _pre('Model solution', r['model_solution']) + _pre('Reference (fallible)', r['reference_solution']) + '</section>'
        for r in stage1]))
    write_text(pkg / 'items_stage2.html', html_page('Stage 2: rate descriptions A/B', [
        f'<section><h2>{html.escape(r["item_code"])} — description {r["description_label"]}</h2>' +
        _pre('Description', r['description_text']) + '</section>' for r in stage2]))
    write_text(pkg / 'pairs.html', html_page('Pair review: same specific observable error?', [
        f'<section><h2>{html.escape(r["pair_code"])}</h2><div class="pair"><div>' +
        _pre('Response 1 question', r['response_1_question']) + _pre('Response 1 solution', r['response_1_solution']) +
        _pre('Response 1 reference', r['response_1_reference']) + '</div><div>' +
        _pre('Response 2 question', r['response_2_question']) + _pre('Response 2 solution', r['response_2_solution']) +
        _pre('Response 2 reference', r['response_2_reference']) + '</div></div></section>' for r in pair_rows]))
    write_json(pkg / 'review_assignment.json', assignment)
    write_jsonl(key_dir / 'item_label_key.jsonl', item_key)
    write_jsonl(key_dir / 'pair_key.jsonl', pair_key)
    write_text(key_dir / 'DO_NOT_SHARE_WITH_REVIEWERS.txt',
               'This directory maps blinded codes to methods, ranks and source IDs. Reviewers must not see it.\n')
    summary = {'items': len(stage1), 'item_descriptions': len(stage2), 'pairs_before_dedup': sum(len(v) for v in pairs.values()),
               'pairs_unique': len(pair_rows), 'max_possible_pairs': len(retrievals) * k,
               'second_review_items': len(second_items), 'second_review_pairs': len(second_pairs),
               'files': log, 'package_sha256': {name: sha256_text((tpl / 'reviewer_1' / name).read_text(encoding='utf-8'))
                                                 for name in ('items_stage1.csv', 'items_stage2.csv', 'pairs.csv')}}
    write_json(pkg / 'package_summary.json', summary)
    return summary


def load_reviews(reviews_dir, pair_codes, item_codes):
    """Read completed reviewer folders. Returns dict(human=..., ai_provisional=..., problems=[])."""
    reviews_dir = Path(reviews_dir)
    if not reviews_dir.is_dir():
        raise ProtocolError(f'Reviews directory not found: {reviews_dir}')
    out = {'human': {}, 'ai_provisional': {}, 'problems': []}
    import json
    for sub in sorted(p for p in reviews_dir.iterdir() if p.is_dir()):
        meta_path = sub / 'reviewer.json'
        if not meta_path.exists():
            out['problems'].append(f'{sub.name}: missing reviewer.json (ignored)')
            continue
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        rtype = meta.get('reviewer_type')
        bucket = 'human' if rtype == 'human' else 'ai_provisional' if rtype == 'ai_provisional' else None
        if bucket is None:
            raise ProtocolError(f'{sub.name}: reviewer_type must be "human" or "ai_provisional"')
        if bucket == 'human' and meta.get('saw_method_key_or_scores'):
            raise ProtocolError(f'{sub.name}: reviewer saw the method key; not independent')
        rid = meta.get('reviewer_id') or sub.name
        entry = {'meta': meta, 'pairs': {}, 'items_stage1': {}, 'items_stage2': {}}
        specs = [('pairs.csv', 'pairs', 'pair_code', None, PAIR_VALUES, pair_codes),
                 ('items_stage1.csv', 'items_stage1', 'item_code', None, ITEM1_VALUES, item_codes),
                 ('items_stage2.csv', 'items_stage2', 'item_code', 'description_label', ITEM2_VALUES, item_codes)]
        for fname, field, code_col, sub_col, allowed, valid in specs:
            path = sub / fname
            if not path.exists():
                continue
            for row in read_csv(path):
                code = row.get(code_col, '').lstrip("'")
                if code not in valid:
                    raise ProtocolError(f'{sub.name}/{fname}: unknown code {code!r}')
                key = (code, row[sub_col]) if sub_col else code
                if key in entry[field]:
                    raise ProtocolError(f'{sub.name}/{fname}: duplicate row for {key}')
                values = {}
                for col, ok in allowed.items():
                    v = (row.get(col) or '').strip().lower()
                    if v and v not in ok:
                        raise ProtocolError(f'{sub.name}/{fname}: {col}={v!r} not in {ok}')
                    values[col] = v or None
                values['free_text'] = {c: row.get(c) for c in row if c in ('observed_error_or_uncertainty', 'reviewer_notes')}
                entry[field][key] = values
        out[bucket][rid] = entry
    return out
