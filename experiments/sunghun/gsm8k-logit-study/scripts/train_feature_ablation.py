"""GPU train-inner-tuned feature ablations; outer validation is comparison only."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

spec = importlib.util.spec_from_file_location('hp', Path(__file__).with_name('train_hidden_probes.py'))
hp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hp)
base = hp.base


def refit(x, y, kind, lr, decay, seed, epochs):
    torch.manual_seed(seed)
    model = hp.build_model(x.shape[1], kind).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
    labels = torch.tensor(y, dtype=torch.float32, device='cuda').reshape(-1, 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((y == 0).sum() / (y == 1).sum()), device='cuda'))
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), labels)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    return model.cpu()


def transform(scaler, values):
    return torch.tensor(np.clip(scaler.transform(values), -10, 10), dtype=torch.float32, device='cuda')


def univariate_scan(frame, y, train, valid):
    rows = []
    for field in frame.columns:
        x = frame[field].to_numpy()
        train_auc = roc_auc_score(y[train], x[train])
        direction = 1 if train_auc >= .5 else -1
        std = x[train].std()
        rows.append({'feature': field, 'train_auroc_oriented': max(train_auc, 1 - train_auc),
            'direction_from_train': direction, 'validation_auroc': roc_auc_score(y[valid], direction * x[valid]),
            'validation_ap': average_precision_score(y[valid], direction * x[valid]),
            'train_correct_mean': float(x[train & (y == 0)].mean()),
            'train_incorrect_mean': float(x[train & (y == 1)].mean()),
            'train_standardized_mean_difference': float((x[train & (y == 1)].mean() - x[train & (y == 0)].mean()) / max(std, 1e-12))})
    return pd.DataFrame(rows).sort_values('train_auroc_oriented', ascending=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    source, output = Path(args.input), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    data = np.load(source / 'designs.npz', allow_pickle=False)
    ids, y, train, valid = [data[key] for key in ('ids', 'y', 'train', 'valid')]
    assert not np.any(train & valid) and np.all(train | valid)
    fit_ids, tune_ids = train_test_split(np.flatnonzero(train), test_size=.2, stratify=y[train], random_state=2026)
    inner_split = {'fit_ids': ids[fit_ids].tolist(), 'tune_ids': ids[tune_ids].tolist(),
                   'train_ids': ids[train].tolist(), 'validation_ids': ids[valid].tolist()}
    base.write_json(output / 'split_ids.json', inner_split)
    groups = [name for name in data.files if name not in ('ids', 'y', 'train', 'valid')]
    protocol = {'test_loaded': False, 'outer_validation_tuning': False, 'groups': groups,
        'primary_checkpoint': 32, 'common_cohort': len(ids), 'train_n': int(train.sum()), 'validation_n': int(valid.sum()),
        'inner_fit_n': len(fit_ids), 'inner_tune_n': len(tune_ids), 'inner_seed': 2026,
        'seeds': hp.SEEDS, 'models': ['linear', 'mlp'], 'mlp_width': 32,
        'grid': {'linear': [[.003, .1], [.003, 1.]], 'mlp': [[.0003, .1], [.001, .1]]},
        'selection': 'Inner-tune ensemble AUROC and early-stop epochs; refit on all outer-train with those fixed epochs.',
        'preprocessing': 'StandardScaler fitted on each training partition; clip +/-10.',
        'comparison': 'Outer validation AUROC/AP and fixed top-20-percent risk budget. No threshold tuning.',
        'limits': ['Exploratory feature comparison, not a new test result.',
            'Original validation was examined in past experiments; not an untouched confirmation set.',
            'No outer-validation choice influences these fitted models, but any subsequent choice still needs new data.',
            'Pointwise bootstrap CIs conditional on fitted models, without multiplicity adjustment.',
            'First-token and t32 vectors refer to different positions and contexts.',
            'Weighted scores are not calibrated error probabilities.'],
        'source_sha256': base.sha(Path(__file__)), 'input_sha256': base.sha(source / 'designs.npz'),
        'device': torch.cuda.get_device_name(), 'torch': torch.__version__}
    base.write_json(output / 'protocol.json', protocol)
    frame = pd.read_parquet(source / 'engineered_features.parquet').loc[ids]
    scan = univariate_scan(frame, y, train, valid)
    scan.to_csv(output / 'univariate_scan.csv', index=False)
    corr = frame.iloc[np.flatnonzero(train)].corr(method='spearman')
    corr.to_parquet(output / 'train_spearman.parquet')
    redundant = [{'a': corr.index[i], 'b': corr.columns[j], 'spearman': corr.iloc[i, j]}
        for i in range(len(corr)) for j in range(i) if abs(corr.iloc[i, j]) >= .95]
    pd.DataFrame(redundant).to_csv(output / 'redundant_pairs.csv', index=False)
    fitted, selections, search = {}, {}, []
    # Outer validation vectors and labels are never passed to model fitting or early stopping.
    for group in groups:
        raw = data[group]
        inner_scaler = StandardScaler().fit(raw[fit_ids])
        x, vx = transform(inner_scaler, raw[fit_ids]), transform(inner_scaler, raw[tune_ids])
        scaler = StandardScaler().fit(raw[train])
        full_x = transform(scaler, raw[train])
        joblib.dump(scaler, output / f'{group}_scaler.joblib')
        for kind in ('linear', 'mlp'):
            best = -np.inf
            key = f'{group}_{kind}'
            for lr, decay in protocol['grid'][kind]:
                pairs = [hp.fit_probe(x, y[fit_ids], vx, y[tune_ids], kind, lr, decay, seed) for seed in hp.SEEDS]
                score = np.mean([p[2] for p in pairs], axis=0)
                auc = float(roc_auc_score(y[tune_ids], score))
                search.append({'key': key, 'lr': lr, 'decay': decay, 'inner_auroc': auc})
                if auc > best:
                    best, chosen = auc, (lr, decay, [p[1] for p in pairs])
                del pairs
            lr, decay, epochs = chosen
            models = [refit(full_x, y[train], kind, lr, decay, seed, epoch) for seed, epoch in zip(hp.SEEDS, epochs)]
            selections[key] = {'group': group, 'model': kind, 'dimensions': raw.shape[1],
                'parameters_per_seed': sum(p.numel() for p in models[0].parameters()),
                'learning_rate': lr, 'weight_decay': decay, 'epochs': epochs, 'inner_auroc': best}
            torch.save([m.state_dict() for m in models], output / f'{key}.pt')
            fitted[key] = (models, scaler)
            print(f'FROZEN {key}: inner AUROC={best:.3f}', flush=True)
    base.write_json(output / 'selections.json', selections)
    base.write_json(output / 'search_history.json', search)
    print('ALL_MODELS_FROZEN; evaluating outer validation', flush=True)
    predictions = pd.DataFrame({'id': ids[valid], 'incorrect': y[valid]})
    rows = []
    for key, (models, scaler) in fitted.items():
        selection = selections[key]
        tx = transform(scaler, data[selection['group']][valid])
        scores = []
        for model in models:
            scores.append(hp.probability(model.cuda(), tx))
            model.cpu()
        score = np.mean(scores, axis=0)
        predictions[key] = score
        k = int(np.ceil(len(score) * .2))
        selected = np.argsort(-score, kind='stable')[:k]
        row = {**selection, 'validation_auroc': float(roc_auc_score(y[valid], score)),
            'validation_ap': float(average_precision_score(y[valid], score)),
            'auroc_ci95': hp.bootstrap_auc(y[valid], score, replicates=1000),
            'top20_precision': float(y[valid][selected].mean()),
            'top20_recall': float(y[valid][selected].sum() / y[valid].sum()),
            'error_prevalence': float(y[valid].mean()), 'validation_n': int(valid.sum()),
            'single_seed_aurocs': [float(roc_auc_score(y[valid], s)) for s in scores]}
        rows.append(row)
        print(f'VALIDATION {key}: AUROC={row["validation_auroc"]:.3f}, AP={row["validation_ap"]:.3f}', flush=True)
    pairs = {}
    for kind in ('linear', 'mlp'):
        for reference, candidate in [('initial_all', 'initial_plus_current'), ('initial_all', 'initial_plus_changes'),
                                     ('raw_initial', 'raw_initial_plus_current'), ('size', 'current_structure'),
                                     ('stability', 'current_all')]:
            pairs[f'{candidate}_vs_{reference}_{kind}'] = base.paired_bootstrap(y[valid],
                predictions[f'{reference}_{kind}'].to_numpy(), predictions[f'{candidate}_{kind}'].to_numpy(), replicates=1000)
    predictions.to_parquet(output / 'validation_predictions.parquet', index=False)
    base.write_json(output / 'results.json', rows)
    pd.DataFrame(rows).to_csv(output / 'results.csv', index=False)
    base.write_json(output / 'paired_comparisons.json', pairs)
    fig, axis = plt.subplots(figsize=(11, 9), layout='constrained')
    for offset, kind, color in ((-.12, 'linear', '#167a68'), (.12, 'mlp', '#c44852')):
        subset = [next(r for r in rows if r['group'] == group and r['model'] == kind) for group in groups]
        auc = np.array([r['validation_auroc'] for r in subset])
        ci = np.array([r['auroc_ci95'] for r in subset]).T
        axis.errorbar(auc, np.arange(len(groups)) + offset, xerr=[auc-ci[0], ci[1]-auc],
                      fmt='o', color=color, capsize=3, markersize=5, label=kind.upper())
    axis.set_yticks(np.arange(len(groups)), groups)
    axis.invert_yaxis()
    axis.axvline(.5, color='#777777', linestyle='--', linewidth=1)
    axis.set_xlim(.3, .95)
    axis.set_xlabel('Outer validation AUROC (200 problems); pointwise 95% CI')
    axis.set_title('Depth-14 feature ablations: training settings selected inside train only')
    axis.legend()
    axis.grid(axis='x', alpha=.2)
    fig.savefig(output / 'feature_ablation.png', dpi=160)
    plt.close(fig)
    lines = ['# Feature Ablation, Train/Validation Only', '',
        'Same t32-active cohort: 799 train, 200 validation. Test is not loaded.',
        'Tune within train (639 fit / 160 tune), then refit on 799 with fixed settings and epochs.',
        'Scores are 3-seed ensembles. AP baseline equals validation error prevalence.', '',
        '| Group | Model | Inputs | Validation AUROC | AP | Top20 precision |',
        '|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["group"]} | {r["model"]} | {r["dimensions"]} | {r["validation_auroc"]:.3f} | '
                     f'{r["validation_ap"]:.3f} | {r["top20_precision"]:.3f} |')
    lines += ['', 'No layer >14, future token or final-layer confidence is a predictor.',
        'Scalar acceleration concerns update magnitudes/directions, not second differences of full hidden vectors.',
        'Temporal features compare different prediction positions: they include changing token/context as well as generation progression.',
        'Many correlated features and group comparisons: this is exploratory, not confirmation.',
        'Outer validation was examined in earlier studies. A winner requires new held-out problems.',
        'No response-regeneration, new residual extraction, or accuracy-improving intervention was performed.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    base.write_json(output / 'complete.json', {'elapsed_seconds': time.monotonic() - started,
        'files': {p.name: base.sha(p) for p in output.iterdir() if p.is_file()}})
    print('FEATURE_ABLATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
