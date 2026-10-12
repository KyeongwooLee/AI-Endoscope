"""Problem-level exploratory error classifiers; original study artifacts stay immutable."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def design(frame, ids):
    metrics = ['entropy', 'margin', 'generated_logprob', 'lower_tail_logprob',
               'js_final', 'adjacent_js', 'top1_agreement', 'log_rank', 'support_change_abs']
    blocks, internal, confidence = [], [], []
    for layer in (7, 14, 21, 27, 28):
        fields = (['entropy', 'margin', 'generated_logprob', 'lower_tail_logprob']
                  if layer == 28 else metrics)
        selected = frame.loc[frame.layer == layer, ['id', 'region', *fields]]
        if selected.duplicated(['id', 'region']).any():
            raise ValueError('Duplicate problem/layer/region')
        pivot = selected.pivot(index='id', columns='region', values=fields)
        pivot.columns = [f'L{layer}_{region}_{feature}' for feature, region in pivot.columns]
        (confidence if layer == 28 else internal).extend(pivot.columns)
        blocks.append(pivot)
    fields = ['stabilization_depth', 'late_stabilization_depth', 'top1_switches']
    selected = frame.loc[frame.layer == 7, ['id', 'region', *fields]]
    pivot = selected.pivot(index='id', columns='region', values=fields)
    pivot.columns = [f'{region}_{feature}' for feature, region in pivot.columns]
    internal.extend(pivot.columns)
    blocks.append(pivot)
    length = frame.groupby('id').generated_tokens.first().rename('generated_tokens')
    result = pd.concat([*blocks, length], axis=1).reindex(ids)
    result['generated_tokens'] = np.log1p(result['generated_tokens'])
    groups = {'length': ['generated_tokens'], 'confidence': confidence,
              'internal': internal, 'combined': confidence + internal,
              'confidence_length': confidence + ['generated_tokens'],
              'combined_length': confidence + internal + ['generated_tokens']}
    return result, groups


def threshold_for_f1(y, scores):
    thresholds = np.unique(np.r_[0., scores, 1.])
    values = [f1_score(y, scores >= threshold, zero_division=0) for threshold in thresholds]
    return float(thresholds[int(np.argmax(values))])


def metrics(y, score, threshold):
    predicted = score >= threshold
    return {'auroc': float(roc_auc_score(y, score)),
            'pr_auc_ap': float(average_precision_score(y, score)),
            'accuracy': float((predicted == y).mean()),
            'balanced_accuracy': float(balanced_accuracy_score(y, predicted)),
            'precision': float(precision_score(y, predicted, zero_division=0)),
            'recall': float(recall_score(y, predicted, zero_division=0)),
            'f1': float(f1_score(y, predicted, zero_division=0)),
            'threshold': float(threshold),
            'confusion_tn_fp_fn_tp': confusion_matrix(y, predicted, labels=[0, 1]).ravel().tolist()}


def fit_mlp(x, y, vx, vy, learning_rate, seed):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(x.shape[1], 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((y == 0).sum()/(y == 1).sum())))
    inputs, labels = torch.from_numpy(x), torch.from_numpy(y.astype(np.float32)).reshape(-1, 1)
    validation = torch.from_numpy(vx)
    best, checkpoint, best_epoch = -np.inf, None, 0
    for epoch in range(1, 301):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite training loss')
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            score = model(validation).sigmoid().numpy().ravel()
        auc = roc_auc_score(vy, score)
        if auc > best + 1e-5:
            best, checkpoint, best_epoch = auc, copy.deepcopy(model.state_dict()), epoch
        if epoch - best_epoch >= 30:
            break
    model.load_state_dict(checkpoint)
    model.eval()
    return model, best_epoch


def probability(model, values):
    if isinstance(model, LogisticRegression):
        return model.predict_proba(values)[:, 1]
    with torch.no_grad():
        return model(torch.from_numpy(values)).sigmoid().numpy().ravel()


def paired_bootstrap(y, reference, candidate, seed=42, replicates=2000):
    rng = np.random.default_rng(seed)
    negative, positive = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    delta_auc, delta_ap = [], []
    for _ in range(replicates):
        indexes = np.r_[rng.choice(negative, len(negative)), rng.choice(positive, len(positive))]
        target = y[indexes]
        delta_auc.append(roc_auc_score(target, candidate[indexes]) - roc_auc_score(target, reference[indexes]))
        delta_ap.append(average_precision_score(target, candidate[indexes]) - average_precision_score(target, reference[indexes]))
    return {'delta_auroc': float(roc_auc_score(y, candidate)-roc_auc_score(y, reference)),
            'delta_auroc_ci95': np.quantile(delta_auc, [.025, .975]).tolist(),
            'delta_ap': float(average_precision_score(y, candidate)-average_precision_score(y, reference)),
            'delta_ap_ci95': np.quantile(delta_ap, [.025, .975]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run, output = Path(args.run).resolve(), Path(args.output).resolve()
    if output.exists():
        raise ValueError('Preserve experiment outputs; use a new output directory')
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    labels = {}
    for phase in ('exploration', 'test'):
        labels[phase] = pd.DataFrame([json.loads(p.read_text(encoding='utf-8'))
                                     for p in sorted((run / 'labels' / phase).glob('*.json'))]).set_index('id')
    all_ids = labels['exploration'].index.to_numpy()
    train_all, validation_all = train_test_split(all_ids, test_size=.2, random_state=42,
        stratify=labels['exploration'].answer_correct.to_numpy())
    eligible = labels['exploration'].analysis_eligible
    train_ids = [i for i in train_all if eligible[i]]
    validation_ids = [i for i in validation_all if eligible[i]]
    test_ids = labels['test'].index[labels['test'].analysis_eligible].tolist()
    assert not set(train_ids) & set(validation_ids)
    assert not (set(train_ids) | set(validation_ids)) & set(test_ids)
    write_json(output / 'split_ids.json', {'train_all': train_all.tolist(),
        'validation_all': validation_all.tolist(), 'train_eligible': train_ids,
        'validation_eligible': validation_ids, 'test_eligible': test_ids})
    protocol = {'positive_class': 'incorrect', 'seed': 42, 'internal_layers': [7, 14, 21, 27],
        'output_layer': 28, 'logistic_C': [.01, .1, 1.], 'mlp_learning_rates': [.001, .003],
        'mlp_seeds': [42, 43, 44], 'mlp_hidden': 32, 'dropout': .2, 'weight_decay': .01,
        'max_epochs': 300, 'early_stop_patience': 30, 'selection': 'validation AUROC',
        'threshold': 'validation max F1', 'bootstrap': 'paired stratified problem bootstrap, 2000',
        'limitations': ['Exploratory follow-up: original test aggregate results were already seen.',
                         'No test-based feature/hyperparameter/threshold selection.',
                         'Weighted classification scores are not calibrated error probabilities.',
                         'Invalid/truncated answers excluded via pre-existing analysis_eligible flag.'],
        'source_sha256': sha(Path(__file__)), 'train_input_sha256': sha(run / 'analysis/exploration/example_features.parquet'),
        'counts': {'train': len(train_ids), 'validation': len(validation_ids), 'test': len(test_ids)},
        'packages': {'python': platform.python_version(), 'torch': torch.__version__, 'sklearn': sklearn.__version__}}
    write_json(output / 'protocol.json', protocol)
    train_frame = pd.read_parquet(run / 'analysis/exploration/example_features.parquet')
    train_design, groups = design(train_frame, [*train_ids, *validation_ids])
    write_json(output / 'feature_groups.json', groups)
    y = (~labels['exploration'].loc[train_ids, 'answer_correct']).to_numpy(dtype=int)
    vy = (~labels['exploration'].loc[validation_ids, 'answer_correct']).to_numpy(dtype=int)
    fitted, selections = {}, {}
    for group, fields in groups.items():
        imputer = SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
        scaler = StandardScaler()
        x = np.clip(scaler.fit_transform(imputer.fit_transform(train_design.loc[train_ids, fields])), -10, 10).astype(np.float32)
        vx = np.clip(scaler.transform(imputer.transform(train_design.loc[validation_ids, fields])), -10, 10).astype(np.float32)
        joblib.dump({'imputer': imputer, 'scaler': scaler, 'fields': fields}, output / f'{group}_preprocessing.joblib')
        for kind in ('logistic', 'mlp'):
            best = -np.inf
            for parameter in (protocol['logistic_C'] if kind == 'logistic' else protocol['mlp_learning_rates']):
                if kind == 'logistic':
                    models = [LogisticRegression(C=parameter, class_weight='balanced', max_iter=3000,
                                                 random_state=42).fit(x, y)]
                    epochs = []
                else:
                    pairs = [fit_mlp(x, y, vx, vy, parameter, seed) for seed in protocol['mlp_seeds']]
                    models, epochs = [p[0] for p in pairs], [p[1] for p in pairs]
                validation_score = np.mean([probability(m, vx) for m in models], axis=0)
                auc = roc_auc_score(vy, validation_score)
                if auc > best:
                    best = auc
                    chosen = (models, epochs, validation_score, parameter)
            models, epochs, score, parameter = chosen
            key = f'{group}_{kind}'
            threshold = threshold_for_f1(vy, score)
            fitted[key] = (models, imputer, scaler, fields, threshold)
            selections[key] = {'group': group, 'model': kind, 'parameter': parameter,
                               'epochs': epochs, 'input_dimensions': x.shape[1],
                               'validation': metrics(vy, score, threshold)}
            if kind == 'logistic':
                joblib.dump(models[0], output / f'{key}.joblib')
            else:
                torch.save([m.state_dict() for m in models], output / f'{key}.pt')
            print(f'Frozen {key}: validation AUROC={best:.3f}', flush=True)
    # Freeze every choice before loading test feature values.
    write_json(output / 'validation_selection.json', selections)
    test_frame = pd.read_parquet(run / 'analysis/test/example_features.parquet')
    test_design, test_groups = design(test_frame, test_ids)
    assert groups == test_groups
    ty = (~labels['test'].loc[test_ids, 'answer_correct']).to_numpy(dtype=int)
    predictions = pd.DataFrame({'id': test_ids, 'incorrect': ty})
    results = []
    for key, (models, imputer, scaler, fields, threshold) in fitted.items():
        tx = np.clip(scaler.transform(imputer.transform(test_design[fields])), -10, 10).astype(np.float32)
        score = np.mean([probability(m, tx) for m in models], axis=0)
        predictions[key] = score
        values = metrics(ty, score, threshold)
        results.append({'group': selections[key]['group'], 'model': selections[key]['model'],
                        'dimensions': selections[key]['input_dimensions'],
                        'validation_auroc': selections[key]['validation']['auroc'], **values})
        print(f'{key}: test AUROC={values["auroc"]:.3f}, AP={values["pr_auc_ap"]:.3f}', flush=True)
    predictions.to_parquet(output / 'test_predictions.parquet', index=False)
    comparisons = {}
    for kind in ('logistic', 'mlp'):
        for suffix in ('', '_length'):
            reference, candidate = f'confidence{suffix}_{kind}', f'combined{suffix}_{kind}'
            comparisons[f'{candidate}_vs_{reference}'] = paired_bootstrap(
                ty, predictions[reference].to_numpy(), predictions[candidate].to_numpy())
    write_json(output / 'paired_comparisons.json', comparisons)
    write_json(output / 'results.json', {'test_error_prevalence': float(ty.mean()),
        'majority_accuracy': float(1-ty.mean()), 'results': results, 'comparisons': comparisons})
    table = pd.DataFrame(results)
    table.to_csv(output / 'results.csv', index=False)
    lines = ['# Error Classifier Follow-Up', '',
        'Exploratory analysis; the original test aggregate was previously inspected. No test tuning.', '',
        f'Eligible train/validation/test: {len(train_ids)}/{len(validation_ids)}/{len(test_ids)}.',
        f'Test error prevalence / random AP baseline: {ty.mean():.4f}.',
        'Incorrect = positive. Invalid/truncated responses excluded; missing feature values imputed from training only.',
        'MLP scores average three seeds. Preprocessing, early stopping, hyperparameters and F1 thresholds use train/validation only.', '',
        '| Features | Model | Val AUROC | Test AUROC | Test AP | Precision | Recall | F1 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in results:
        lines.append(f'| {row["group"]} | {row["model"]} | {row["validation_auroc"]:.3f} | {row["auroc"]:.3f} | {row["pr_auc_ap"]:.3f} | {row["precision"]:.3f} | {row["recall"]:.3f} | {row["f1"]:.3f} |')
    lines += ['', '## Added Internal Information', '', 'Paired 95% bootstrap intervals, stratified by outcome; no multiple-testing correction.']
    for name, value in comparisons.items():
        lines.append(f'- {name}: AUROC difference {value["delta_auroc"]:.4f}, CI {value["delta_auroc_ci95"]}; AP difference {value["delta_ap"]:.4f}, CI {value["delta_ap_ci95"]}.')
    lines += ['', 'These models detect associations, not causal error mechanisms. A classifier failure does not establish absence of internal information.',
              'Raw hidden states/full logits are not used. Length controls are not question-difficulty controls.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(output / 'complete.json', {'completed_unix': time.time(),
        'files': {p.name: sha(p) for p in output.iterdir() if p.is_file() and p.name != 'complete.json'}})
    print('Saved report: ' + str(output / 'report.md'), flush=True)


if __name__ == '__main__':
    main()
