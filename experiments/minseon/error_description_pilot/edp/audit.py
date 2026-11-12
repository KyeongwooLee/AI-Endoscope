"""Source fetching, checksum verification, schema validation and audit counts."""
import collections
import csv
import io
import json
import re
import subprocess
import urllib.request
from pathlib import Path

from .io import (ProtocolError, is_finite_number, now_utc, question_sha256, read_json,
                 read_jsonl, sha256_file, sha256_text)

RULE_TYPES = {'SETUP', 'NOEQ', 'TARGET', 'OTHER', 'ARITH', 'NEWNUM'}
REQUIRED_FIELDS = {
    'id': str, 'split': str, 'pid': int, 'sample': int, 'mode': str, 'question': str,
    'reference': str, 'gold': (int, float), 'pred': (int, float), 'response': str,
    'finish': str, 'has_correct_sibling': bool, 'rule_type': str,
}
PINNED_BRANCHES = {
    'main': '51560936fa89329e481a2c8c8f7af223df27576a',
    'dahyeon': '51560936fa89329e481a2c8c8f7af223df27576a',
    'minseon': '73d02635b55358c2b942c2bcd084ebb334c0db85',
    'minsu': 'd95116f0d8a6b64a7f3eb08457af11144d6f4a7c',
    'sunghun': '99b35512ba043b0ec86d809ec700fcb4ba969889',
    'kyeongwoo': '7c4485dbc3f14e3771210fe626efe155e8eb468e',
}
REPO_URL = 'https://github.com/KyeongwooLee/AI-Endoscope'


def _download(url, target, expected):
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + '.download')
    request = urllib.request.Request(url, headers={'User-Agent': 'edp-pilot'})
    with urllib.request.urlopen(request, timeout=120) as response, temp.open('wb') as handle:
        while block := response.read(1 << 20):
            handle.write(block)
    actual = sha256_file(temp)
    if actual != expected:
        raise ProtocolError(f'Download checksum mismatch for {url}: {actual} != {expected}; kept {temp}')
    temp.replace(target)


def fetch_and_verify(cfg, data_dir, allow_download=True):
    """Ensure every pinned asset exists and matches its SHA256 before parsing."""
    manifest = {}
    for key, source in cfg['sources'].items():
        if not isinstance(source, dict) or 'assets' not in source:
            continue
        for name, expected in source['assets'].items():
            path = Path(data_dir) / source['subdir'] / name
            url = source['base_url'] + source.get('asset_urls', {}).get(name, name)
            if not path.exists():
                if not allow_download:
                    raise ProtocolError(f'Missing asset {path}')
                _download(url, path, expected)
            actual = sha256_file(path)
            if actual != expected:
                raise ProtocolError(f'Checksum mismatch, preserve and investigate: {path} {actual} != {expected}')
            manifest[f'{source["release"]}/{name}'] = {
                'source_key': key, 'url': url, 'path': str(path), 'bytes': path.stat().st_size,
                'sha256': actual, 'verified': True}
    return manifest


def validate_rows(rows, meta_ids):
    """Strict schema checks; returns problems (raises on fatal ones)."""
    ids = [r.get('id') for r in rows]
    if len(set(ids)) != len(ids):
        dup = [k for k, v in collections.Counter(ids).items() if v > 1][:5]
        raise ProtocolError(f'Duplicate source IDs: {dup}')
    if meta_ids != ids:
        raise ProtocolError('features_meta.json ids do not match errors.jsonl row order')
    for index, row in enumerate(rows):
        for field, kind in REQUIRED_FIELDS.items():
            if field not in row:
                raise ProtocolError(f'Row {index} missing {field}')
            value = row[field]
            if kind is int and (type(value) is not int):
                raise ProtocolError(f'Row {index} field {field} not int')
            if not isinstance(value, kind) or (kind is not bool and isinstance(value, bool)):
                raise ProtocolError(f'Row {index} field {field} has type {type(value).__name__}')
        if row['split'] != 'train':
            raise ProtocolError(f'Row {index} unexpected split {row["split"]}')
        if row['id'] != f'train-{row["pid"]}-{row["sample"]}':
            raise ProtocolError(f'Row {index} id/pid/sample mismatch: {row["id"]}')
        if row['mode'] != ('greedy' if row['sample'] == 0 else 'sampled') or not 0 <= row['sample'] <= 4:
            raise ProtocolError(f'Row {index} inconsistent mode/sample')
        if row['finish'] not in ('stop', 'length'):
            raise ProtocolError(f'Row {index} unknown finish {row["finish"]}')
        if not (is_finite_number(row['gold']) and is_finite_number(row['pred'])):
            raise ProtocolError(f'Row {index} non-finite gold/pred')
        if row['rule_type'] not in RULE_TYPES:
            raise ProtocolError(f'Row {index} unknown rule_type')
        if abs(row['gold'] - row['pred']) < 1e-6:
            raise ProtocolError(f'Row {index} is numerically correct under the release rule')


def build_source_records(rows, release, asset_sha):
    """Derived identifiers for every source row (source files stay immutable)."""
    first = {}
    records = []
    for index, row in enumerate(rows):
        key = (row['pid'], row['response'])
        canonical = first.setdefault(key, row['id'])
        records.append({
            'source_release': release, 'source_asset_sha256': asset_sha, 'source_row_index': index,
            'source_id': row['id'], 'namespaced_id': f'{release}:{row["id"]}', 'source_pid': row['pid'],
            'question_key': f'gsm8k:train:{row["pid"]}', 'question_sha256': question_sha256(row['question']),
            'response_sha256': sha256_text(row['response']), 'canonical_response_id': canonical,
            'source_mode': row['mode'], 'source_finish': row['finish'],
        })
    return records


def audit_counts(rows, records):
    by_pid_question = collections.defaultdict(set)
    by_qhash_pids = collections.defaultdict(set)
    by_response_pids = collections.defaultdict(set)
    for row, rec in zip(rows, records):
        by_pid_question[row['pid']].add((row['question'], row['reference'], row['gold']))
        by_qhash_pids[rec['question_sha256']].add(row['pid'])
        by_response_pids[rec['response_sha256']].add(row['pid'])
    inconsistent = [pid for pid, values in by_pid_question.items() if len(values) > 1]
    if inconsistent:
        raise ProtocolError(f'pids with inconsistent question/reference/gold: {inconsistent[:5]}')
    pair_counts = collections.Counter((r['pid'], r['response']) for r in rows)
    return {
        'responses': len(rows), 'unique_response_ids': len({r['id'] for r in rows}),
        'questions': len(by_pid_question),
        'sampled': sum(r['mode'] == 'sampled' for r in rows), 'greedy': sum(r['mode'] == 'greedy' for r in rows),
        'finish_stop': sum(r['finish'] == 'stop' for r in rows),
        'finish_length': sum(r['finish'] == 'length' for r in rows),
        'extra_identical_pid_response_copies': sum(c - 1 for c in pair_counts.values()),
        'identical_response_text_across_pids': sum(len(v) > 1 for v in by_response_pids.values()),
        'normalized_question_groups_spanning_multiple_pids': sum(len(v) > 1 for v in by_qhash_pids.values()),
        'rule_counts': dict(collections.Counter(r['rule_type'] for r in rows).most_common()),
        'finish_length_with_valid_final_marker': sum(r['finish'] == 'length' and '####' in r['response'] for r in rows),
        'empty_or_whitespace_responses': sum(not r['response'].strip() for r in rows),
    }


def _read_tsv_ids(path):
    text = Path(path).read_text(encoding='utf-8')
    return [row['id'] for row in csv.DictReader(io.StringIO(text), delimiter='\t')]


def prior_overlap(cfg, data_dir, rows, records):
    """Question-level overlap with prior annotation/explanation artifacts."""
    qhash_to_pid = {rec['question_sha256']: rec['source_pid'] for rec in records}
    pid_to_qhash = {rec['source_pid']: rec['question_sha256'] for rec in records}
    pair_set = {(rec['question_sha256'], rec['response_sha256']) for rec in records}
    excluded = {}  # question_key -> list of reasons
    report = {'checked': [], 'not_checked': []}

    def mark(pid, reason):
        excluded.setdefault(f'gsm8k:train:{pid}', set()).add(reason)

    # 1. Sunghun's 1,100 AI-draft annotations (separate responses; question-level only).
    src = cfg['sources']['prior_annotations']
    labels = read_jsonl(Path(data_dir) / src['subdir'] / 'labels.jsonl')
    label_q = set()
    exact = 0
    hash_field_mismatch = 0
    for row in labels:
        qh = question_sha256(row['question'])
        rh = sha256_text(row['response'])
        if row.get('response_sha256') and row['response_sha256'] != rh:
            hash_field_mismatch += 1
        if qh in qhash_to_pid:
            label_q.add(qh)
            mark(qhash_to_pid[qh], 'prior_annotation_question:error-taxonomy-full-v1-draft')
        if (qh, rh) in pair_set:
            exact += 1
    overlap_rows = sum(rec['question_sha256'] in label_q for rec in records)
    report['error_taxonomy_full_v1_draft'] = {
        'annotation_rows': len(labels), 'datasets': dict(collections.Counter(r.get('dataset') for r in labels)),
        'overlap_questions': len(label_q), 'overlap_primary_rows': overlap_rows,
        'exact_normalized_question_and_unchanged_response_pairs': exact,
        'stored_response_sha256_disagreements': hash_field_mismatch,
        'match_rule': 'NFC + whitespace-collapsed question SHA256; exact response UTF-8 SHA256',
        'note': 'AI drafts, human-unreviewed; legacy explanations are never passed to annotators.'}
    report['checked'].append('error-taxonomy-full-v1-draft/labels.jsonl')

    # 2. Minsu's AI-review rubric outputs on this release (train-{pid}-{sample} IDs).
    src = cfg['sources']['minsu_ai_review']
    by_id = {r['id']: r for r in rows}
    minsu = {}
    for name in src['assets']:
        ids = _read_tsv_ids(Path(data_dir) / src['subdir'] / name)
        known = [i for i in ids if i in by_id]
        unknown = [i for i in ids if i not in by_id]
        pids = {by_id[i]['pid'] for i in known}
        for pid in pids:
            mark(pid, f'prior_ai_review_question:minsu:{name}')
        minsu[name] = {'rows': len(ids), 'matched_primary_ids': len(known), 'unmatched_ids': len(unknown),
                       'unmatched_examples': unknown[:5], 'questions': len(pids)}
        report['checked'].append(f'minsu@d95116f:{name}')
    report['minsu_ai_review'] = minsu

    # 3. Local Minseon artifacts (manual adjudications / shown examples) -> verified by question text.
    local_questions = {}
    for directory in cfg['sources'].get('local_prior_records_dirs', []):
        d = Path(directory)
        if not d.is_dir():
            report['not_checked'].append(f'missing local records dir: {directory}')
            continue
        for path in sorted(d.glob('gsm8k_train_*.json')):
            try:
                obj = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            local_questions[obj.get('id', path.stem)] = obj.get('question', '')
        report['checked'].append(f'local records (question text only): {directory}')
    local_hashes = {question_sha256(q) for q in local_questions.values() if q}
    report['local_generation_records'] = {
        'train_records': len(local_questions),
        'questions_overlapping_primary': len(local_hashes & set(qhash_to_pid)),
        'note': 'Different Qwen generations used for A/B detection/repair; not error annotations. Informational only.'}
    local = {}
    for artifact in cfg['sources'].get('local_prior_artifacts', []):
        p = Path(artifact)
        if not p.exists():
            report['not_checked'].append(f'missing local artifact: {artifact}')
            continue
        text = p.read_text(encoding='utf-8', errors='replace')
        found = sorted(set(re.findall(r'gsm8k_train_(\d+)', text)), key=int)
        verified, unverified = [], []
        for n in found:
            q = local_questions.get(f'gsm8k_train_{n}')
            qh = question_sha256(q) if q else None
            if qh and qh in qhash_to_pid:
                verified.append(qhash_to_pid[qh])
                mark(qhash_to_pid[qh], f'local_reviewed_example_question:{p.name}')
            elif q is None:
                unverified.append(n)
        local[str(p)] = {'train_ids_found': len(found), 'matched_primary_questions': len(verified),
                         'pid_index_consistent': all(pid_to_qhash.get(int(n)) == question_sha256(local_questions[f'gsm8k_train_{n}'])
                                                     for n in found if f'gsm8k_train_{n}' in local_questions and int(n) in pid_to_qhash),
                         'unverified_ids_without_question_text': unverified}
        report['checked'].append(f'local artifact: {artifact}')
    report['local_reviewed_artifacts'] = local
    report['not_checked'].extend([
        'Team-discussion explanation attempts not present as runnable artifacts in the inspected branches.',
        'Dahyeon annotation work (branch tip equals main; not reconstructable).',
        "Sunghun's qwen-extra-errors / posthoc-transfer releases (different responses; not used as corpus).",
    ])
    ledger = {k: sorted(v) for k, v in sorted(excluded.items())}
    report['excluded_question_groups_total'] = len(ledger)
    report['excluded_question_groups_by_reason'] = dict(collections.Counter(
        reason.split(':')[0] + ':' + reason.split(':')[1] for v in ledger.values() for reason in v))
    return report, ledger


def branch_heads():
    try:
        out = subprocess.run(['git', 'ls-remote', REPO_URL], capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return {'checked_at': now_utc(), 'error': str(exc)}
    heads = {}
    for line in out.splitlines():
        sha, ref = line.split('\t')
        if ref.startswith('refs/heads/'):
            heads[ref[len('refs/heads/'):]] = sha
    return {'checked_at': now_utc(), 'observed': heads,
            'pinned': PINNED_BRANCHES,
            'changed_since_pin': {b: heads.get(b) for b, s in PINNED_BRANCHES.items() if heads.get(b) != s}}


def load_primary(cfg, data_dir):
    src = cfg['sources']['primary']
    base = Path(data_dir) / src['subdir']
    rows = read_jsonl(base / 'errors.jsonl')
    meta = read_json(base / 'features_meta.json')
    validate_rows(rows, meta['ids'])
    records = build_source_records(rows, src['release'], src['assets']['errors.jsonl'])
    return rows, records
