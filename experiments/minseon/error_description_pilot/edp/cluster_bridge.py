"""Development-only K-means diagnostic and the ID-aligned SAE bridge."""
import collections
import csv
from pathlib import Path

import numpy as np

from .io import ProtocolError, read_json, write_jsonl


def kmeans_diagnostic(vectors, ids, k_values, seeds, n_init):
    """Every tried K/seed reported; illustrative only (no K is 'selected')."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    x = np.asarray(vectors, dtype=np.float64)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    runs = []
    for k in k_values:
        labels_by_seed = {}
        for seed in seeds:
            if k >= len(x):
                runs.append({'k': k, 'seed': seed, 'status': 'skipped_k_ge_n'})
                continue
            model = KMeans(n_clusters=k, random_state=seed, n_init=n_init).fit(x)
            sizes = sorted(collections.Counter(model.labels_.tolist()).values(), reverse=True)
            sil = float(silhouette_score(x, model.labels_)) if len(set(model.labels_)) > 1 else None
            labels_by_seed[seed] = model.labels_
            runs.append({'k': k, 'seed': seed, 'sizes': sizes, 'singleton_clusters': sum(s == 1 for s in sizes),
                         'degenerate': min(sizes) <= 1, 'silhouette_descriptive': sil,
                         'assignments': dict(zip(ids, model.labels_.tolist()))})
        if len(labels_by_seed) > 1:
            s = sorted(labels_by_seed)
            runs.append({'k': k, 'seed_ari': [float(adjusted_rand_score(labels_by_seed[a], labels_by_seed[b]))
                                              for i, a in enumerate(s) for b in s[i + 1:]]})
    return runs


def write_bridge(path, manifest, overlays, review_status):
    rows = []
    for m in manifest:
        rows.append({
            'source_release': m['source_release'], 'source_asset_sha256': m['source_asset_sha256'],
            'source_row_index': m['source_row_index'], 'source_id': m['source_id'], 'source_pid': m['source_pid'],
            'question_key': m['question_key'], 'response_sha256': m['response_sha256'],
            'canonical_response_id': m['canonical_response_id'], 'all_source_ids': m['all_source_ids'],
            'sae_partition': m['sae_partition'], 'pilot_role': m['pilot_role'],
            'E_status': (overlays['E'].get(m['source_id']) or {}).get('status', 'missing'),
            'G_status': (overlays['G'].get(m['source_id']) or {}).get('status', 'missing'),
            'human_review_status': review_status.get(m['source_id'], 'pending'),
            'sae_features_attached': False})
    write_jsonl(path, rows)
    return len(rows)


def check_sae_run(results_dir, split_by_id, manifest):
    """Validate a provided Kyeongwoo SAE output dir against our split/IDs; never fabricate features."""
    if not results_dir:
        return {'status': 'no_sae_run_provided'}
    d = Path(results_dir)
    report = {'results_dir': str(d)}
    if not (d / 'splits.json').exists():
        return {**report, 'status': 'invalid_missing_splits_json'}
    theirs = read_json(d / 'splits.json')
    mismatch = [i for i, p in split_by_id.items() if theirs.get(i) != p]
    report['split_mismatches'] = len(mismatch)
    if mismatch:
        raise ProtocolError(f'SAE splits differ from ours for {len(mismatch)} IDs, e.g. {mismatch[:3]}')
    extraction = read_json(d / 'extraction.json') if (d / 'extraction.json').exists() else {}
    report['extraction_complete'] = bool(extraction.get('complete'))
    assign = d / 'report' / 'assignments.csv'
    if assign.exists():
        with assign.open(encoding='utf-8', newline='') as h:
            rows = {r['id']: r for r in csv.DictReader(h)}
        pilot = [m['source_id'] for m in manifest]
        report['pilot_ids_in_assignments'] = sum(i in rows for i in pilot)
    report['status'] = 'complete_run_validated' if report['extraction_complete'] else 'partial_run_not_used'
    return report
