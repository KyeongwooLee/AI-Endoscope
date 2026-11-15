from collections import Counter
from itertools import combinations
from pathlib import Path

from . import extract, label, sae, taxonomy
from .common import digest, lock, object_hash, prepared, read_json, responses, runtime, save, space


def transformed(train, test, components, seed):
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    # Every fitted statistic, including zero-variance removal, sees discovery only.
    keep = np.flatnonzero(np.var(train, axis=0) > 1e-12)
    if not len(keep):
        raise ValueError('No varying discovery features')
    scaler = StandardScaler().fit(train[:, keep])
    x, z = scaler.transform(train[:, keep]), scaler.transform(test[:, keep])
    count = min(components, len(train) - 1, len(keep))
    if count < 1:
        raise ValueError('Too few discovery observations')
    pca = PCA(n_components=count, svd_solver='randomized', random_state=seed).fit(x)
    return pca.transform(x), pca.transform(z), {'scaler': scaler, 'pca': pca, 'keep': keep}


def bootstrap_metrics(y, score, groups, runs, seed):
    import numpy as np
    from sklearn.metrics import average_precision_score, f1_score
    y, score, groups = np.asarray(y), np.asarray(score), np.asarray(groups)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    estimates = []
    indices = {g: np.flatnonzero(groups == g) for g in unique}
    for _ in range(runs):
        pick = np.concatenate([indices[g] for g in rng.choice(unique, len(unique), replace=True)])
        if len(np.unique(y[pick])) != 2:
            continue
        estimates.append([average_precision_score(y[pick], score[pick]), f1_score(y[pick], score[pick] >= .5, zero_division=0)])
    ci = np.quantile(estimates, [.025, .975], axis=0).T.tolist() if len(estimates) >= max(10, runs // 2) else [None, None]
    return {'auprc': float(average_precision_score(y, score)),
        'f1': float(f1_score(y, score >= .5, zero_division=0)), 'auprc_ci95': ci[0], 'f1_ci95': ci[1],
        'bootstrap_valid_runs': len(estimates), 'bootstrap_requested_runs': runs}


def support(y, groups):
    import numpy as np
    return {'responses': len(y), 'positive_responses': int(sum(y)),
        'positive_problems': len(set(np.asarray(groups)[np.asarray(y) == 1])),
        'negative_problems': len(set(np.asarray(groups)[np.asarray(y) == 0])), 'problems': len(set(groups))}


def paired_differences(x, records, annotations, left, right):
    import numpy as np
    groups = {}
    for index, row in enumerate(records):
        types = annotations[row['id']]['label']['types']
        a, b = left in types, right in types
        if a != b:
            groups.setdefault(row['pid'], {left: [], right: []})[left if a else right].append(index)
    result = [x[g[left]].mean(0) - x[g[right]].mean(0) for g in groups.values() if g[left] and g[right]]
    return np.stack(result) if result else np.empty((0, x.shape[1]), dtype=np.float32)


def contrasts(x, records, annotations, type_ids, cfg):
    import numpy as np
    discovery = [i for i,r in enumerate(records) if r['partition'] == 'discovery']
    confirmation = [i for i,r in enumerate(records) if r['partition'] == 'confirmation']
    result = []
    settings = cfg['analysis']
    for left, right in combinations(type_ids, 2):
        train = paired_differences(x[discovery], [records[i] for i in discovery], annotations, left, right)
        test = paired_differences(x[confirmation], [records[i] for i in confirmation], annotations, left, right)
        record = {'left': left, 'right': right, 'discovery_paired_problems': len(train),
                  'confirmation_paired_problems': len(test)}
        if len(train) < settings['min_train_positive_problems'] or len(test) < settings['min_test_positive_problems']:
            result.append({**record, 'status': 'insufficient_paired_problems'})
            continue
        top = np.argsort(np.abs(train.mean(0)))[-settings['top_features']:][::-1]
        means = test[:, top].mean(0)
        rng = np.random.default_rng(cfg['seed'])
        boots = [test[rng.integers(0, len(test), len(test))][:, top].mean(0) for _ in range(settings['bootstrap_runs'])]
        bounds = np.quantile(boots, [.025, .975], axis=0)
        result.append({**record, 'status': 'ok', 'features_selected_on': 'discovery',
            'features': [{'feature': int(f), 'discovery_mean_difference': float(train[:, f].mean()),
                'confirmation_mean_difference': float(means[j]),
                'confirmation_problem_bootstrap_ci95': [float(bounds[0,j]), float(bounds[1,j])]}
                for j, f in enumerate(top)]})
    return result


def feature_examples(arrays, records, quality, y, top):
    import numpy as np
    by_id = {r['id']: r for r in quality}
    result = []
    confirmation = [i for i,r in enumerate(records) if r['partition'] == 'confirmation' and y[i]]
    for feature in top:
        if not confirmation:
            continue
        best = max(confirmation, key=lambda i: arrays['sae_max'][i, feature])
        row = records[best]
        token = int(arrays['peaks'][best, feature])
        offsets = by_id[row['id']].get('offsets', [])
        if token < 0 or token >= len(offsets):
            continue
        start, end = offsets[token]
        if start < 0:
            continue
        result.append({'feature': int(feature), 'id': row['id'], 'pid': row['pid'],
            'partition': 'confirmation', 'token_index': token, 'char_span': [start, end],
            'peak_activation': float(arrays['sae_max'][best, feature]),
            'token_text': row['response'][start:end], 'context': row['response'][max(0,start-80):end+80]})
    return result


def evaluate_entry(cfg, entry, records_by_id, annotations):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    ids, arrays, quality, state = extract.load_entry(cfg, entry)
    if state['labels_sha256'] != object_hash(list(annotations.values())):
        raise ValueError('Features were extracted for a different label snapshot')
    indices = [i for i, response_id in enumerate(ids) if annotations[response_id]['label']['status'] == 'classified']
    records = [records_by_id[ids[i]] for i in indices]
    arrays = {k: v[indices] for k,v in arrays.items()}
    if not records:
        return {'status': 'no_classified_eligible_features', 'types': [], 'paired': [], 'quality': summarize_quality(quality), 'extraction': state}
    discovery = np.array([i for i,r in enumerate(records) if r['partition'] == 'discovery'], dtype=int)
    confirmation = np.array([i for i,r in enumerate(records) if r['partition'] == 'confirmation'], dtype=int)
    type_ids = [c['id'] for c in taxonomy.frozen(cfg)['codebook']['categories']]
    settings = cfg['analysis']
    bases = {'sae': np.concatenate([arrays[k] for k in ('sae_mean', 'sae_max', 'sae_frequency')], axis=1),
             'raw_hidden': arrays['hidden_mean'],
             'length': np.array([[len(r['output_token_ids'])] for r in records], dtype=np.float32),
             'correctness': np.array([[int(r['answer_correct'])] for r in records], dtype=np.float32)}
    bases['length_correctness'] = np.concatenate([bases['length'], bases['correctness']], axis=1)
    fitted, exclusions = {}, {}
    if len(discovery) >= 2 and len(confirmation):
        for name, x in bases.items():
            try:
                if name in ('sae', 'raw_hidden'):
                    train, test, _ = transformed(x[discovery], x[confirmation], settings['pca_components'], cfg['seed'])
                else:
                    scaler = StandardScaler().fit(x[discovery])
                    train, test = scaler.transform(x[discovery]), scaler.transform(x[confirmation])
                fitted[name] = (train, test)
            except ValueError as exc:
                exclusions[name] = str(exc)
    metrics, common = [], Counter()
    for kind in type_ids:
        y = np.array([kind in annotations[r['id']]['label']['types'] for r in records], dtype=int)
        groups = np.array([r['pid'] for r in records])
        train_support, test_support = support(y[discovery], groups[discovery]), support(y[confirmation], groups[confirmation])
        record = {'type': kind, 'discovery': train_support, 'confirmation': test_support,
                  'baselines': {}, 'baseline_exclusions': exclusions}
        if (min(train_support['positive_problems'], train_support['negative_problems']) < settings['min_train_positive_problems']
                or min(test_support['positive_problems'], test_support['negative_problems']) < settings['min_test_positive_problems']):
            metrics.append({**record, 'status': 'insufficient_positive_or_negative_problems'})
            continue
        for name, (x,z) in fitted.items():
            classifier = LogisticRegression(C=1., max_iter=2000, random_state=cfg['seed']).fit(x, y[discovery])
            score = classifier.predict_proba(z)[:, 1]
            record['baselines'][name] = bootstrap_metrics(y[confirmation], score, groups[confirmation],
                settings['bootstrap_runs'], cfg['seed'])
        mean = arrays['sae_mean'][discovery]
        delta = (mean[y[discovery] == 1].mean(0) - mean[y[discovery] == 0].mean(0)) / np.maximum(mean.std(0), 1e-6)
        top = np.argsort(np.abs(delta))[-settings['top_features']:][::-1]
        common.update(int(f) for f in top)
        record.update(status='ok', positive_prevalence=float(y[confirmation].mean()),
            features_selected_on='discovery', top_features=[{'feature': int(f), 'standardized_mean_difference': float(delta[f])} for f in top],
            max_activation_examples=feature_examples(arrays, records, quality, y, top))
        metrics.append(record)
    return {'status': 'ok', 'types': metrics, 'quality': summarize_quality(quality), 'extraction': state,
        'paired': contrasts(arrays['sae_mean'], records, annotations, type_ids, cfg),
        'common_features': [{'feature': f, 'types_in_top_features': n} for f,n in common.most_common() if n > 1],
        'feature_policy': 'same labeled, classified, token-eligible IDs for all five baselines; SAE mean/max/frequency concatenated before PCA',
        'classifier': 'StandardScaler + discovery PCA + L2 logistic C=1; F1 threshold fixed at 0.5; no confirmation fitting'}


def summarize_quality(quality):
    import numpy as np
    ok = [r for r in quality if r['status'] == 'ok']
    return {'responses': len(quality), 'eligible_responses': len(ok),
        'excluded_reasons': dict(Counter(r.get('reason') for r in quality if r['status'] != 'ok')),
        **{key: float(np.mean([r[key] for r in ok])) if ok else None
            for key in ('reconstruction_mse', 'relative_squared_error', 'mean_l0', 'active_fraction')}}


def run(cfg):
    annotations_list = label.load(cfg)
    annotations = {r['id']: r for r in annotations_list}
    if not annotations:
        raise ValueError('No labels; label and validate a problem sample before analysis')
    records = {r['id']: r for r in responses(cfg) if r['id'] in annotations}
    if not set(annotations) <= set(records):
        raise ValueError('Labels refer to unknown responses')
    out = Path(cfg['run_dir']) / 'analysis'
    with lock(out):
        registry = sae.registry(cfg)
        signature = object_hash({'labels': annotations_list, 'codebook': taxonomy.frozen(cfg),
            'config': cfg, 'registry': registry, 'runtime': runtime(),
            'extraction_index': read_json(Path(cfg['run_dir']) / 'extraction/index.json')})
        results = {}
        for entry in registry['entries']:
            if entry['status'] != 'metadata_verified':
                continue
            target = out / (entry['id'] + '.json')
            if target.exists():
                value = read_json(target)
                if value['signature'] != signature:
                    raise ValueError('Analysis inputs changed; preserve previous results')
            else:
                space(cfg, 64 * 2**20)
                value = {'signature': signature, 'entry': entry,
                    **evaluate_entry(cfg, entry, records, annotations)}
                save(target, value)
            results[entry['id']] = digest(target)
            print(f'Analyzed {entry["id"]}', flush=True)
        save(out / 'index.json', {'signature': signature, 'files': results})
