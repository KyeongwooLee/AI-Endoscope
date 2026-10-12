"""GPU probes on fixed, pre-final-layer residual vectors; validation-only selection."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

spec = importlib.util.spec_from_file_location('base_classifier', Path(__file__).with_name('train_error_classifier.py'))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
CHECKPOINTS = (0, 32, 64, 128)
VARIANTS = ('h14', 'h14_delta13', 'h7_h10_h14')
SEEDS = (42, 43, 44)


def load_phase(data, phase):
    completion = json.loads((data / 'complete.json').read_text())
    ids, targets, checkpoints, hidden = [], [], [], []
    for entry in completion['batches']:
        file = data / entry['file']
        if base.sha(file) != entry['sha256']:
            raise ValueError(f'Hash mismatch: {file}')
        with np.load(file, allow_pickle=False) as batch:
            mask = batch['phases'] == phase
            if not mask.any():
                continue
            ids.extend(batch['ids'][mask].tolist())
            targets.extend(batch['y'][mask].tolist())
            checkpoints.extend(batch['checkpoints'][mask].tolist())
            hidden.append(batch['hidden'][mask])
    result = {'ids': np.asarray(ids), 'y': np.asarray(targets),
              'checkpoints': np.asarray(checkpoints), 'hidden': np.concatenate(hidden)}
    assert np.isfinite(result['hidden']).all()
    assert len(set(zip(ids, checkpoints))) == len(ids)
    return result


def make_features(hidden, variant):
    # Stored ordering is layers 7, 10, 13, 14. No layer >14 is accessed.
    if variant == 'h14':
        return hidden[:, 3]
    if variant == 'h14_delta13':
        return np.concatenate((hidden[:, 3], hidden[:, 3] - hidden[:, 2]), axis=1)
    if variant == 'h7_h10_h14':
        return hidden[:, (0, 1, 3)].reshape(len(hidden), -1)
    raise ValueError(variant)


def build_model(dimensions, kind):
    if kind == 'linear':
        return nn.Linear(dimensions, 1)
    return nn.Sequential(nn.Linear(dimensions, 32), nn.ReLU(), nn.Dropout(.2), nn.Linear(32, 1))


def probability(model, x):
    model.eval()
    with torch.inference_mode():
        return model(x).sigmoid().cpu().numpy().ravel()


def fit_probe(x, y, vx, vy, kind, learning_rate, weight_decay, seed):
    torch.manual_seed(seed)
    model = build_model(x.shape[1], kind).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    labels = torch.tensor(y, dtype=torch.float32, device='cuda').reshape(-1, 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((y == 0).sum() / (y == 1).sum(), device='cuda'))
    best, best_epoch, saved = -np.inf, 0, None
    for epoch in range(1, 301):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), labels)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite loss')
        loss.backward()
        optimizer.step()
        score = probability(model, vx)
        auc = roc_auc_score(vy, score)
        if auc > best + 1e-5:
            best, best_epoch = auc, epoch
            saved = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch - best_epoch >= 30:
            break
    model.load_state_dict(saved)
    return model, best_epoch, probability(model, vx)


def bootstrap_auc(y, score, replicates=2000):
    rng = np.random.default_rng(42)
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    values = []
    for _ in range(replicates):
        selected = np.r_[rng.choice(positive, len(positive)), rng.choice(negative, len(negative))]
        values.append(roc_auc_score(y[selected], score[selected]))
    return np.quantile(values, [.025, .975]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--splits', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    data, output = Path(args.data), Path(args.output)
    if output.exists():
        raise ValueError('Use a new results directory')
    output.mkdir(parents=True)
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    splits = json.loads(Path(args.splits).read_text())
    train_set, valid_set = set(splits['train_all']), set(splits['validation_all'])
    assert len(train_set) == 800 and len(valid_set) == 200 and not train_set & valid_set
    protocol = {'primary_checkpoint': 32, 'secondary_checkpoints': [0, 64, 128],
        'variants': VARIANTS, 'models': ['linear', 'mlp'], 'mlp_width': 32,
        'seeds': SEEDS, 'max_epochs': 300, 'patience': 30,
        'selection': 'Validation AUROC of mean of three seeds; max validation F1 threshold.',
        'preprocessing': 'Train-only StandardScaler, clip +/-10, no PCA or test-derived transforms.',
        'target': 'Incorrect final answer, including parse failures and truncations.',
        'inputs': 'Current prediction position only; h7/h10/h13/h14 raw residual vectors.',
        'forbidden': ['layer >14', 'chosen next token', 'future positions', 'whole answer length', 'gold answer'],
        'limitations': ['Exploratory: previous test results have already been inspected.',
            'Later checkpoints restrict population to still-active generations; do not compare as identical cohorts.',
            'Offline causal replay, not live detection or an accuracy-improving intervention.',
            'Weighted scores are not calibrated probabilities.',
            'Three-seed ensemble is reported separately from single-seed probes.',
            'Same width but input dimensions and parameter counts differ across variants.'],
        'data_config_sha256': base.sha(data / 'config.json'),
        'split_sha256': base.sha(Path(args.splits)), 'source_sha256': base.sha(Path(__file__)),
        'base_source_sha256': base.sha(Path(__file__).with_name('train_error_classifier.py')),
        'device': torch.cuda.get_device_name(), 'torch': torch.__version__}
    base.write_json(output / 'protocol.json', protocol)
    exploration = load_phase(data, 'exploration')
    fitted, selections, population, history = {}, {}, {}, []
    for checkpoint in CHECKPOINTS:
        available = exploration['checkpoints'] == checkpoint
        ids, y, hidden = [exploration[key][available] for key in ('ids', 'y', 'hidden')]
        train = np.array([identifier in train_set for identifier in ids])
        valid = np.array([identifier in valid_set for identifier in ids])
        assert (train | valid).all() and not (train & valid).any()
        population[checkpoint] = {'train_ids': ids[train].tolist(), 'validation_ids': ids[valid].tolist()}
        for variant in VARIANTS:
            raw = make_features(hidden, variant)
            scaler = StandardScaler().fit(raw[train])
            x = torch.tensor(np.clip(scaler.transform(raw[train]), -10, 10), dtype=torch.float32, device='cuda')
            vx = torch.tensor(np.clip(scaler.transform(raw[valid]), -10, 10), dtype=torch.float32, device='cuda')
            joblib.dump(scaler, output / f't{checkpoint}_{variant}_scaler.joblib')
            for kind in ('linear', 'mlp'):
                grid = ([(.003, wd) for wd in (.01, .1, 1.)] if kind == 'linear'
                        else [(lr, wd) for lr in (.0003, .001) for wd in (.1, 1.)])
                best = -np.inf
                key = f't{checkpoint}_{variant}_{kind}'
                for learning_rate, weight_decay in grid:
                    pairs = [fit_probe(x, y[train], vx, y[valid], kind, learning_rate, weight_decay, seed) for seed in SEEDS]
                    score = np.mean([pair[2] for pair in pairs], axis=0)
                    auc = roc_auc_score(y[valid], score)
                    history.append({'key': key, 'learning_rate': learning_rate,
                                    'weight_decay': weight_decay, 'validation_auroc': float(auc)})
                    if auc > best:
                        best = auc
                        chosen = ([copy.deepcopy(pair[0]).cpu() for pair in pairs],
                                  [pair[1] for pair in pairs], score, learning_rate, weight_decay)
                    del pairs
                models, epochs, score, lr, wd = chosen
                threshold = base.threshold_for_f1(y[valid], score)
                selection = {'checkpoint': checkpoint, 'variant': variant, 'model': kind,
                    'dimensions': raw.shape[1], 'parameters_per_seed': sum(p.numel() for p in models[0].parameters()),
                    'learning_rate': lr, 'weight_decay': wd, 'epochs': epochs,
                    'validation': base.metrics(y[valid], score, threshold),
                    'train_n': int(train.sum()), 'validation_n': int(valid.sum())}
                selections[key] = selection
                torch.save({'state_dicts': [m.state_dict() for m in models], 'selection': selection}, output / f'{key}.pt')
                fitted[key] = (models, scaler, threshold)
                print(f'FROZEN {key}: validation AUROC={best:.4f}', flush=True)
    base.write_json(output / 'validation_selections.json', selections)
    base.write_json(output / 'selection_history.json', history)
    base.write_json(output / 'populations.json', population)
    print('ALL_SELECTIONS_FROZEN; loading test inputs', flush=True)
    test = load_phase(data, 'test')
    assert not set(test['ids']) & (train_set | valid_set)
    rows, predictions, paired = [], [], {}
    for checkpoint in CHECKPOINTS:
        available = test['checkpoints'] == checkpoint
        ids, y, hidden = [test[key][available] for key in ('ids', 'y', 'hidden')]
        prediction = pd.DataFrame({'id': ids, 'incorrect': y})
        for variant in VARIANTS:
            raw = make_features(hidden, variant)
            for kind in ('linear', 'mlp'):
                key = f't{checkpoint}_{variant}_{kind}'
                models, scaler, threshold = fitted[key]
                tx = torch.tensor(np.clip(scaler.transform(raw), -10, 10), dtype=torch.float32, device='cuda')
                scores = []
                for seed, model in zip(SEEDS, models):
                    score = probability(model.cuda(), tx)
                    model.cpu()
                    scores.append(score)
                    prediction[f'{variant}_{kind}_seed{seed}'] = score
                score = np.mean(scores, axis=0)
                prediction[f'{variant}_{kind}'] = score
                row = {key2: value for key2, value in selections[key].items() if key2 != 'validation'}
                row.update(base.metrics(y, score, threshold))
                row.update({'test_n': len(y), 'coverage': len(y) / 1319,
                    'error_prevalence': float(y.mean()), 'majority_accuracy': float(max(y.mean(), 1-y.mean())),
                    'validation_auroc': selections[key]['validation']['auroc'],
                    'single_seed_aurocs': [float(roc_auc_score(y, s)) for s in scores]})
                if checkpoint == 32:
                    row['auroc_ci95'] = bootstrap_auc(y, score)
                rows.append(row)
                print(f'TEST {key}: AUROC={row["auroc"]:.4f}, AP={row["pr_auc_ap"]:.4f}', flush=True)
        if checkpoint == 32:
            for kind in ('linear', 'mlp'):
                for variant in VARIANTS[1:]:
                    paired[f'{variant}_vs_h14_{kind}'] = base.paired_bootstrap(y,
                        prediction[f'h14_{kind}'].to_numpy(), prediction[f'{variant}_{kind}'].to_numpy())
        prediction.to_parquet(output / f'predictions_t{checkpoint}.parquet', index=False)
    base.write_json(output / 'results.json', rows)
    pd.DataFrame(rows).to_csv(output / 'results.csv', index=False)
    base.write_json(output / 'paired_comparisons.json', paired)
    report = ['# Raw Hidden-State Error Probes', '',
        'Primary checkpoint: after 32 generated tokens, at raw decoder block 14.',
        'Positive class: incorrect final answer. GPU training; no Qwen weight updates.',
        'Scores below are three-seed ensembles. Single-seed AUROCs are retained in results.json.', '',
        '| Input | Model | Validation AUROC | Test AUROC (95% CI) | AP | Precision | Recall | Parameters/seed |',
        '|---|---|---:|---|---:|---:|---:|---:|']
    for row in rows:
        if row['checkpoint'] == 32:
            lo, hi = row['auroc_ci95']
            report.append(f'| {row["variant"]} | {row["model"]} | {row["validation_auroc"]:.3f} | '
                f'{row["auroc"]:.3f} [{lo:.3f}, {hi:.3f}] | {row["pr_auc_ap"]:.3f} | '
                f'{row["precision"]:.3f} | {row["recall"]:.3f} | {row["parameters_per_seed"]:,} |')
    report += ['', 'Thresholds selected on validation max F1. AP baseline is the test error prevalence.',
        'Confidence intervals: 2,000 stratified problem bootstrap samples, conditional on the trained models.',
        'No multiple-comparison adjustment; all findings remain exploratory.',
        'h14 + (h14-h13) and concatenated h13,h14 carry the same information; the delta representation changes the inductive bias.',
        'Offline risk discrimination does not establish increased answer accuracy or calibrated probabilities.',
        'See paired_comparisons.json for whether additional layers improve over h14 within the same model class.']
    (output / 'probe_report.md').write_text('\n'.join(report) + '\n')
    base.write_json(output / 'probe_complete.json', {'elapsed_seconds': time.monotonic() - started,
        'models': len(rows), 'files': {str(p.relative_to(output)): base.sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    print('PROBE_TRAINING_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
