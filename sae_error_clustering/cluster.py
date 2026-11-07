import itertools
import os

from .common import atomic_json, atomic_npz, digest, paths, read_json
from .extract import load_features


def transform_features(values, discovery, components, sparse):
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize
    active = np.any(values[discovery] != 0, axis=0)
    if not active.any():
        raise ValueError('No active discovery features')
    x = values[:, active].astype(np.float32)
    if sparse:
        if (x < 0).any():
            raise ValueError('Negative standard SAE activation')
        x = np.log1p(x)
    pca = PCA(n_components=min(components, len(discovery) - 1, x.shape[1]),
              svd_solver='randomized', whiten=False, random_state=42)
    pca.fit(x[discovery])
    coordinates = normalize(pca.transform(x)).astype(np.float32)
    return coordinates, {'active': active, 'pca': pca, 'log1p': sparse}


def score(x, labels, maximum):
    import numpy as np
    from sklearn.metrics import silhouette_score
    if not 1 < len(np.unique(labels)) < len(labels):
        return None
    # Include every cluster in the fixed scoring sample, including rare clusters.
    rng = np.random.default_rng(42)
    mandatory = [np.flatnonzero(labels == c)[0] for c in np.unique(labels)]
    rest = np.setdiff1d(np.arange(len(labels)), mandatory)
    size = min(maximum, len(labels))
    indices = np.r_[mandatory, rng.permutation(rest)[:max(0, size - len(mandatory))]]
    return float(silhouette_score(x[indices], labels[indices]))


def select_k(metrics):
    eligible = [m for m in metrics if m['mean_silhouette'] is not None]
    if not eligible:
        raise ValueError('No nondegenerate cluster candidate; inspect extraction/features')
    return min(eligible, key=lambda m: (-m['mean_silhouette'], m['k']))['k']


def analyze(x, discovery, pids, cfg):
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    candidates, fitted = [], {}
    for k in cfg['k_values']:
        if k >= len(discovery):
            continue
        labels, scores = [], []
        for seed in cfg['seeds']:
            model = KMeans(n_clusters=k, random_state=seed, n_init=cfg['n_init']).fit(x[discovery])
            labels.append(model.labels_)
            scores.append(score(x[discovery], model.labels_, cfg['silhouette_samples']))
            if seed == 42:
                fitted[k] = model
        pairs = [float(adjusted_rand_score(a, b)) for a, b in itertools.combinations(labels, 2)]
        candidates.append({'k': k, 'silhouettes': scores,
                           'mean_silhouette': float(np.mean(scores)) if all(s is not None for s in scores) else None,
                           'seed_ari': pairs,
                           'sizes': [int(v) for v in np.bincount(labels[0], minlength=k)]})
    k = select_k(candidates)
    model = fitted[k]
    resampling = []
    unique = np.unique(pids[discovery])
    for seed in range(42, 42 + cfg['resample_runs']):
        groups = np.random.default_rng(seed).choice(unique, max(1, int(len(unique) * .8)), replace=False)
        subset = discovery[np.isin(pids[discovery], groups)]
        if len(subset) <= k:
            resampling.append(None)
            continue
        fitted_sample = KMeans(n_clusters=k, random_state=seed, n_init=cfg['n_init']).fit(x[subset])
        resampling.append(float(adjusted_rand_score(model.labels_, fitted_sample.predict(x[discovery]))))
    return model, {'selected_k': k, 'candidates': candidates, 'problem_resampling_ari': resampling,
                   'interpretation': 'Exploratory geometric groups, not validated error categories'}


def run(cfg):
    import joblib
    import numpy as np
    rows, arrays, details = load_features(cfg)
    used = np.array([i for i, d in enumerate(details) if d['status'] == 'ok'], dtype=int)
    if len(used) < 4:
        raise ValueError('Too few usable completions')
    valid = [rows[i] for i in used]
    # Choose canonical *usable* responses; excluded copies cannot remove valid data.
    seen = set()
    discovery = []
    for i, row in enumerate(valid):
        key = (row['pid'], row['response'])
        if row['partition'] == 'discovery' and key not in seen:
            discovery.append(i)
            seen.add(key)
    discovery = np.asarray(discovery)
    if len(discovery) < 4:
        raise ValueError('Too few unique discovery completions')
    pids = np.asarray([r['pid'] for r in valid])
    outputs = {'used_indices': used, 'discovery_indices': discovery}
    bundles, metrics = {}, {}
    for label, key, sparse in [('sae', 'sae_mean', True), ('hidden', 'hidden_mean', False)]:
        x, preprocessor = transform_features(arrays[key][used], discovery, cfg['components'], sparse)
        model, metrics[label] = analyze(x, discovery, pids, cfg)
        predicted = model.predict(x)
        distances = model.transform(x)
        confirmation = np.array([i for i, r in enumerate(valid) if r['partition'] == 'confirmation'], dtype=int)
        metrics[label]['confirmation_silhouette'] = score(x[confirmation], predicted[confirmation], cfg['silhouette_samples']) if len(confirmation) > 2 else None
        metrics[label]['all_sizes'] = np.bincount(predicted, minlength=model.n_clusters).tolist()
        metrics[label]['unique_problems'] = [len(set(pids[predicted == c].tolist())) for c in range(model.n_clusters)]
        outputs.update({f'{label}_coordinates': x, f'{label}_labels': predicted, f'{label}_distances': distances})
        bundles[label] = {'preprocessor': preprocessor, 'kmeans': model}
    _, out = paths(cfg)
    temp = out / 'models.joblib.tmp'
    joblib.dump(bundles, temp)
    os.replace(temp, out / 'models.joblib')
    atomic_npz(out / 'clusters.npz', **outputs)
    atomic_json(out / 'metrics.json', metrics)
    atomic_json(out / 'clustering.json', {'extraction': read_json(out / 'extraction.json'),
                'analysis_code_sha256': digest(__file__),
                'clusters_sha256': digest(out / 'clusters.npz'), 'metrics_sha256': digest(out / 'metrics.json'),
                'models_sha256': digest(out / 'models.joblib')})
    print('Clustered discovery data; confirmation data was transformed without refitting.')
