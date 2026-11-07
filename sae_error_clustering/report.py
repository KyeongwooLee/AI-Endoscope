import csv
import html
import os
from collections import Counter

from .common import digest, paths, read_json, read_rows
from .extract import load_features


def write_csv(path, rows, fields):
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def choose_cases(indices, records, distances, cluster):
    import numpy as np
    rng = np.random.default_rng(42 + cluster)
    selected, used_pids = [], set()
    sorted_distances = np.sort(distances, axis=1)
    margin = sorted_distances[:, 1] - sorted_distances[:, 0]
    for kind, order in [('center', sorted(indices, key=lambda i: distances[i, cluster])),
                        ('boundary', sorted(indices, key=lambda i: margin[i])),
                        ('random', rng.permutation(indices).tolist())]:
        count = 0
        for i in order:
            if records[i]['pid'] not in used_pids:
                selected.append((kind, int(i)))
                used_pids.add(records[i]['pid'])
                count += 1
                if count == 5:
                    break
    return selected


def run(cfg):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data, out = paths(cfg)
    provenance = read_json(out / 'clustering.json')
    if provenance['extraction'] != read_json(out / 'extraction.json'):
        raise ValueError('Extraction changed after clustering')
    for name, key in [('clusters.npz', 'clusters_sha256'), ('metrics.json', 'metrics_sha256'), ('models.joblib', 'models_sha256')]:
        if digest(out / name) != provenance[key]:
            raise ValueError('Clustering artifact checksum mismatch')
    records, arrays, details = load_features(cfg)
    with np.load(out / 'clusters.npz', allow_pickle=False) as file:
        clusters = {k: file[k].copy() for k in file.files}
    used = clusters['used_indices']
    rows = [records[i] for i in used]
    labels, distances = clusters['sae_labels'], clusters['sae_distances']
    metrics = read_json(out / 'metrics.json')
    # Rule labels and confidence are read only here, after cluster selection.
    for name in ['errors.jsonl', 'features_meta.json', 'confidence.npy']:
        expected = read_json(out / 'manifest.json')['assets'][name]
        if digest(data / name) != expected:
            raise ValueError(f'Report source checksum mismatch: {name}')
    source = {r['id']: r for r in read_rows(data / 'errors.jsonl')}
    confidence = np.load(data / 'confidence.npy', allow_pickle=False)
    confidence_names = read_json(data / 'features_meta.json')['confidence']
    if confidence.shape != (len(records), len(confidence_names)) or not np.isfinite(confidence).all():
        raise ValueError('Invalid confidence array')
    report_dir = out / 'report'
    report_dir.mkdir(exist_ok=True)
    review_path = report_dir / 'review.csv'
    if review_path.exists():
        with review_path.open(encoding='utf-8-sig', newline='') as handle:
            edited = any(r.get('category_candidate', '').strip() or r.get('notes', '').strip()
                         or r.get('review_status', 'unreviewed') != 'unreviewed'
                         for r in csv.DictReader(handle))
        if edited:
            raise ValueError('Human review.csv contains edits; preserve it elsewhere before regenerating the report')
    assignment = {int(index): int(labels[j]) for j, index in enumerate(used)}
    write_csv(report_dir / 'assignments.csv', [dict(id=r['id'], pid=r['pid'], partition=r['partition'],
        cluster=assignment.get(i, ''), status=details[i]['status'], reason=details[i].get('reason', ''))
        for i, r in enumerate(records)], ['id', 'pid', 'partition', 'cluster', 'status', 'reason'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, label in zip(axes, ['sae', 'hidden']):
        xy = clusters[f'{label}_coordinates']
        ax.scatter(xy[:, 0], xy[:, 1], c=clusters[f'{label}_labels'], s=4, cmap='tab10')
        ax.set(title=f'{label}: first two normalized PCA coordinates (display only)', xlabel='PC1', ylabel='PC2')
    fig.tight_layout()
    fig.savefig(report_dir / 'pca.png', dpi=150)
    plt.close(fig)
    review, feature_rows, cross_rows, summary_rows = [], [], [], []
    text = ['# Exploratory SAE clusters', '',
            'These are post-generation groups of wrong answers, not validated error types or causal mechanisms.',
            'Rule labels are unreviewed automatic annotations. No normal completions were evaluated.',
            'K was selected using discovery geometry only. Confirmation was never used for fitting.',
            f'Processed: {len(records)}; usable: {len(used)}; excluded: {len(records)-len(used)}.',
            '', '![PCA display](pca.png)', '', '## Geometry and stability', '',
            'See ../metrics.json for all K candidates, seed ARI, problem-resampling ARI and confirmation scores.',
            'Low silhouette/stability means the categories may not be supported; a chosen K alone is not evidence.', '']
    text.extend(['| Representation | Selected K | Confirmation silhouette | Problem resampling ARI |',
                 '| --- | --- | --- | --- |'])
    for name in ['sae', 'hidden']:
        m = metrics[name]
        text.append(f'| {name} | {m["selected_k"]} | {m["confirmation_silhouette"]} | {m["problem_resampling_ari"]} |')
    text.extend(['', '| Representation | K | Mean discovery silhouette | Seed ARI |', '| --- | --- | --- | --- |'])
    for name in ['sae', 'hidden']:
        for m in metrics[name]['candidates']:
            text.append(f'| {name} | {m["k"]} | {m["mean_silhouette"]} | {m["seed_ari"]} |')
    text.append('')
    sae = arrays['sae_mean'][used]
    discovery = clusters['discovery_indices']
    discovery_set = set(discovery.tolist())
    for c in range(metrics['sae']['selected_k']):
        members = np.flatnonzero(labels == c)
        train_members = np.array([i for i in members if i in discovery_set], dtype=int)
        other = discovery[labels[discovery] != c]
        text.extend([f'## cluster_{c}', '', f'{len(members)} responses / {len({rows[i]["pid"] for i in members})} problems.',
                     'Category candidate: **unreviewed**. Mark mixed/undetermined when evidence is insufficient.', ''])
        if not len(train_members) or not len(other):
            continue
        frequency_gap = (sae[train_members] > 0).mean(0) - (sae[other] > 0).mean(0)
        mean_gap = sae[train_members].mean(0) - sae[other].mean(0)
        top = sorted(np.flatnonzero(frequency_gap > 0), key=lambda f: (-frequency_gap[f], -mean_gap[f], f))[:20]
        for f in top:
            examples = sorted(train_members, key=lambda i: -sae[i, f])[:3]
            for i in examples:
                original = int(used[i])
                peak = int(arrays['peaks'][original, f])
                if peak < 0:
                    continue
                a, b = details[original]['offsets'][peak]
                snippet = rows[i]['response'][max(0, a-80):b+80]
                feature_rows.append(dict(cluster=c, feature=int(f), frequency_gap=float(frequency_gap[f]),
                    mean_gap=float(mean_gap[f]), id=rows[i]['id'], output_token_index=peak,
                    char_start=a, char_end=b, snippet=snippet))
        chosen = choose_cases(train_members.tolist(), rows, distances, c)
        confirmation = [int(i) for i in members if rows[i]['partition'] == 'confirmation']
        seen = set()
        for i in np.random.default_rng(42+c).permutation(confirmation):
            if rows[i]['pid'] not in seen and len(seen) < 5:
                chosen.append(('confirmation_random', int(i)))
                seen.add(rows[i]['pid'])
        for kind, i in chosen:
            r = rows[i]
            review.append(dict(cluster=c, selection=kind, id=r['id'], pid=r['pid'], partition=r['partition'],
                               question=source[r['id']]['question'], response=r['response'],
                               category_candidate='', review_status='unreviewed', notes=''))
            # HTML escaping prevents model text from injecting report markup.
            text.extend([f'### {kind}: {r["id"]}', '', '<pre>' + html.escape(source[r['id']]['question']) + '</pre>',
                         '<pre>' + html.escape(r['response']) + '</pre>', ''])
        counts = Counter(source[rows[i]['id']]['rule_type'] for i in members)
        cross_rows.extend(dict(cluster=c, rule_type=k, count=v) for k, v in sorted(counts.items()))
        summary = {'cluster': c, 'responses': len(members), 'problems': len({rows[i]['pid'] for i in members})}
        for mode in ['greedy', 'sampled']:
            summary[mode] = sum(rows[i]['mode'] == mode for i in members)
        for j, name in enumerate(confidence_names):
            summary[name + '_median'] = float(np.median(confidence[used[members], j]))
        summary_rows.append(summary)
        text.extend(['Automatic rule-label distribution (post-hoc only): ' + str(dict(counts)), ''])
    write_csv(report_dir / 'review.csv', review, ['cluster', 'selection', 'id', 'pid', 'partition', 'question', 'response', 'category_candidate', 'review_status', 'notes'])
    write_csv(report_dir / 'features.csv', feature_rows, ['cluster', 'feature', 'frequency_gap', 'mean_gap', 'id', 'output_token_index', 'char_start', 'char_end', 'snippet'])
    write_csv(report_dir / 'rule_cross_table.csv', cross_rows, ['cluster', 'rule_type', 'count'])
    write_csv(report_dir / 'cluster_summary.csv', summary_rows, list(summary_rows[0]) if summary_rows else ['cluster'])
    write_csv(report_dir / 'extraction_quality.csv', details,
              ['id', 'status', 'reason', 'tokens', 'active_features', 'reconstruction_mse', 'relative_squared_error'])
    text.extend(['## Review notes', '', 'Use review.csv for human category candidates, including mixed/undetermined.',
                 'Keep edited review files separately: regenerating this report replaces generated templates.',
                 'No classifier, early-error prediction, steering, or correction has been evaluated.',
                 'SAE base-model revision is unspecified in its metadata; numerical compatibility needs review.'])
    temp = report_dir / 'REPORT.md.tmp'
    temp.write_text('\n'.join(text), encoding='utf-8')
    os.replace(temp, report_dir / 'REPORT.md')
    print(f'Report: {report_dir / "REPORT.md"}')
