"""Causal prefix/depth-constrained error probes from saved token-level readouts."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch

spec = importlib.util.spec_from_file_location('base_classifier', Path(__file__).with_name('train_error_classifier.py'))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

CHECKPOINTS = (0, 32, 64, 128)
SAFE_COLUMNS = ['id', 'layer', 'token_index', 'entropy', 'margin', 'adjacent_js']


def load_readouts(run, phase):
    folders = sorted((run / 'features' / phase).iterdir())
    dataset = ds.dataset([str(p / 'token_features.parquet') for p in folders], format='parquet')
    frame = dataset.to_table(columns=SAFE_COLUMNS,
        filter=ds.field('layer').isin([6, 7, 9, 10, 13, 14]) & (ds.field('token_index') <= 128)).to_pandas()
    del dataset
    dataset = ds.dataset([str(p / 'topk_tokens.parquet') for p in folders], format='parquet')
    batches = []
    for batch in dataset.scanner(columns=['id', 'layer', 'token_index', 'top5_ids'],
        filter=(ds.field('layer') <= 14) & (ds.field('token_index') <= 128), batch_size=32768).to_batches():
        table = batch.to_pandas()[['id', 'layer', 'token_index']]
        table['top1'] = pc.list_element(batch.column('top5_ids'), 0).to_numpy()
        batches.append(table)
    top = pd.concat(batches, ignore_index=True)
    return frame, top


def aggregate(series, name, values):
    series = np.asarray(series, dtype=float)
    values[name + '_current'] = float(series[-1])
    values[name + '_prefix_mean'] = float(series.mean())
    values[name + '_prefix_std'] = float(series.std())


def prefix_design(readouts, top, checkpoint):
    rows, simple_fields = [], None
    top_groups = dict(tuple(top.groupby('id', sort=False)))
    for sample, group in readouts.groupby('id', sort=False):
        # An available index t means inference reached the prediction after t prior tokens.
        # Completion before t is excluded as an at-risk population, never a predictor.
        if not ((group.layer == 14) & (group.token_index == checkpoint)).any():
            continue
        prefix = group[group.token_index <= checkpoint]
        values = {'id': sample}
        for metric in ('entropy', 'margin'):
            history = prefix[prefix.layer == 14].sort_values('token_index')[metric]
            aggregate(history, f'L14_{metric}', values)
        simple_fields = [key for key in values if key != 'id']
        for layer in (7, 10):
            for metric in ('entropy', 'margin'):
                history = prefix[prefix.layer == layer].sort_values('token_index')[metric]
                aggregate(history, f'L{layer}_{metric}', values)
        # Saved adjacent_js at row l compares l and l+1, NOT l-1 and l.
        # Thus row 14 would leak layer 15 and is deliberately never used.
        for lower in (6, 9, 13):
            history = prefix[prefix.layer == lower].sort_values('token_index').adjacent_js
            aggregate(history, f'JS_L{lower}_L{lower+1}', values)
        tp = top_groups[sample]
        matrix = tp[tp.token_index <= checkpoint].pivot(index='token_index', columns='layer', values='top1')
        assert matrix.columns.max() == 14 and matrix.index.max() == checkpoint
        assert matrix.shape == (checkpoint + 1, 14)
        candidates = matrix.to_numpy()
        aggregate((candidates[:, 1:] != candidates[:, :-1]).mean(axis=1), 'depth_switch_fraction', values)
        aggregate(candidates[:, 13] == candidates[:, 12], 'L13_L14_agreement', values)
        aggregate(candidates[:, 13] == candidates[:, 6], 'L7_L14_agreement', values)
        values['L14_candidate_prefix_change_rate'] = (float((candidates[1:, 13] != candidates[:-1, 13]).mean())
                                                     if checkpoint else 0.)
        rows.append(values)
    result = pd.DataFrame(rows).set_index('id').sort_index()
    return result, {'mid14': simple_fields, 'mid_multi': result.columns.tolist()}


def bootstrap_auc(y, scores, replicates=2000):
    rng = np.random.default_rng(42)
    a, b = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    values = []
    for _ in range(replicates):
        indexes = np.r_[rng.choice(a, len(a)), rng.choice(b, len(b))]
        values.append(roc_auc_score(y[indexes], scores[indexes]))
    return np.quantile(values, [.025, .975]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--splits', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run, output = Path(args.run).resolve(), Path(args.output).resolve()
    if output.exists():
        raise ValueError('Use a new output directory')
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    splits = json.loads(Path(args.splits).read_text())
    protocol = {'primary_checkpoint': 32, 'secondary_checkpoints': [0, 64, 128],
        'max_layer': 14, 'allowed_readouts': SAFE_COLUMNS,
        'top1_candidate_layers': list(range(1, 15)),
        'time_definition': 'At layer 14 predicting token t+1, using only positions 0..t; no actual token t+1.',
        'forbidden': ['layer>14', 'JS-to-final', 'agreement-to-final', 'generated-token support/rank',
                      'full-depth stabilization', 'whole response length', 'future token positions',
                      'final-answer/reasoning region membership'],
        'saved_adjacent_js_semantics': 'row l compares l to l+1; use rows 6/9/13 only',
        'selection': 'Train/validation only; same problem split as previous classifier',
        'target': 'Final answer_correct flag inverted; parse failures included, no eligibility filtering',
        'population': 'Only runs still predicting at checkpoint; coverage reported separately',
        'limits': ['Exploratory: original test summaries and earlier classifier results were seen.',
                   'Offline causal replay on original full-model-generated prefixes, not an intervention.',
                   'Predicting risk does not establish accuracy improvement.',
                   'LM-head readout cost is not benchmarked; small probe does not imply cheap features.'],
        'source_sha256': base.sha(Path(__file__)),
        'base_training_sha256': base.sha(Path(__file__).with_name('train_error_classifier.py'))}
    base.write_json(output / 'protocol.json', protocol)
    labels = {}
    for phase in ('exploration', 'test'):
        labels[phase] = pd.DataFrame([json.loads(p.read_text(encoding='utf-8')) for p in
                                     sorted((run / 'labels' / phase).glob('*.json'))]).set_index('id')
    print('Loading only depth-safe readout columns from exploration...', flush=True)
    train_readouts, train_top = load_readouts(run, 'exploration')
    designs, groups, fitted, selections = {}, {}, {}, {}
    populations = {}
    for checkpoint in CHECKPOINTS:
        design, mapping = prefix_design(train_readouts, train_top, checkpoint)
        designs[checkpoint], groups[checkpoint] = design, mapping
        train_ids = [i for i in splits['train_all'] if i in design.index]
        valid_ids = [i for i in splits['validation_all'] if i in design.index]
        assert not set(train_ids) & set(valid_ids)
        y = (~labels['exploration'].loc[train_ids, 'answer_correct']).to_numpy(dtype=int)
        vy = (~labels['exploration'].loc[valid_ids, 'answer_correct']).to_numpy(dtype=int)
        populations[checkpoint] = {'train_ids': train_ids, 'validation_ids': valid_ids,
                                   'train_count': len(train_ids), 'validation_count': len(valid_ids)}
        design.to_parquet(output / f'train_prefix_{checkpoint}.parquet')
        for group, fields in mapping.items():
            imputer = SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
            scaler = StandardScaler()
            x = np.clip(scaler.fit_transform(imputer.fit_transform(design.loc[train_ids, fields])), -10, 10).astype(np.float32)
            vx = np.clip(scaler.transform(imputer.transform(design.loc[valid_ids, fields])), -10, 10).astype(np.float32)
            joblib.dump({'imputer': imputer, 'scaler': scaler, 'fields': fields}, output / f't{checkpoint}_{group}_preprocess.joblib')
            for kind in ('logistic', 'mlp'):
                best = -np.inf
                for parameter in ([.01, .1, 1.] if kind == 'logistic' else [.001, .003]):
                    if kind == 'logistic':
                        models = [LogisticRegression(C=parameter, class_weight='balanced',
                                                     max_iter=3000, random_state=42).fit(x, y)]
                        epochs = []
                    else:
                        pairs = [base.fit_mlp(x, y, vx, vy, parameter, seed) for seed in (42, 43, 44)]
                        models, epochs = [p[0] for p in pairs], [p[1] for p in pairs]
                    score = np.mean([base.probability(m, vx) for m in models], axis=0)
                    auc = roc_auc_score(vy, score)
                    if auc > best:
                        best, chosen = auc, (models, epochs, score, parameter)
                models, epochs, score, parameter = chosen
                threshold = base.threshold_for_f1(vy, score)
                key = f't{checkpoint}_{group}_{kind}'
                fitted[key] = (models, imputer, scaler, fields, threshold)
                selections[key] = {'checkpoint': checkpoint, 'group': group, 'model': kind,
                    'parameter': parameter, 'epochs': epochs, 'dimensions': x.shape[1],
                    'validation': base.metrics(vy, score, threshold)}
                if kind == 'logistic':
                    joblib.dump(models[0], output / f'{key}.joblib')
                else:
                    torch.save([m.state_dict() for m in models], output / f'{key}.pt')
                print(f'Frozen {key}: val AUROC={best:.3f}', flush=True)
    base.write_json(output / 'validation_selections.json', selections)
    base.write_json(output / 'feature_groups.json', groups)
    del train_readouts, train_top
    print('All choices frozen. Loading test readouts...', flush=True)
    readouts, top = load_readouts(run, 'test')
    results, comparisons = [], {}
    for checkpoint in CHECKPOINTS:
        design, mapping = prefix_design(readouts, top, checkpoint)
        assert mapping == groups[checkpoint]
        test_ids = design.index.tolist()
        assert not set(test_ids) & set(splits['train_all'] + splits['validation_all'])
        y = (~labels['test'].loc[test_ids, 'answer_correct']).to_numpy(dtype=int)
        populations[checkpoint].update(test_ids=test_ids, test_count=len(test_ids),
            test_coverage=len(test_ids)/len(labels['test']), test_error_prevalence=float(y.mean()))
        design.to_parquet(output / f'test_prefix_{checkpoint}.parquet')
        scores = pd.DataFrame({'id': test_ids, 'incorrect': y})
        for key, selected in selections.items():
            if selected['checkpoint'] != checkpoint:
                continue
            models, imputer, scaler, fields, threshold = fitted[key]
            tx = np.clip(scaler.transform(imputer.transform(design[fields])), -10, 10).astype(np.float32)
            score = np.mean([base.probability(m, tx) for m in models], axis=0)
            scores[key] = score
            values = base.metrics(y, score, threshold)
            row = {'checkpoint': checkpoint, 'group': selected['group'], 'model': selected['model'],
                'dimensions': selected['dimensions'], 'validation_auroc': selected['validation']['auroc'],
                'test_count': len(y), 'test_coverage': len(y)/len(labels['test']),
                'test_error_prevalence': float(y.mean()), **values}
            if checkpoint == protocol['primary_checkpoint']:
                row['auroc_ci95'] = bootstrap_auc(y, score)
            results.append(row)
            print(f'{key}: test AUROC={values["auroc"]:.3f}, AP={values["pr_auc_ap"]:.3f}, coverage={len(y)}/{len(labels["test"])}', flush=True)
        scores.to_parquet(output / f'predictions_t{checkpoint}.parquet', index=False)
        if checkpoint == protocol['primary_checkpoint']:
            for kind in ('logistic', 'mlp'):
                reference, candidate = f't{checkpoint}_mid14_{kind}', f't{checkpoint}_mid_multi_{kind}'
                comparisons[kind] = base.paired_bootstrap(y, scores[reference].to_numpy(), scores[candidate].to_numpy())
    base.write_json(output / 'populations.json', populations)
    base.write_json(output / 'results.json', {'results': results, 'primary_comparisons': comparisons})
    pd.DataFrame(results).to_csv(output / 'results.csv', index=False)
    lines = ['# Mid-Layer Causal-Prefix Classifier', '',
        'Exploratory offline probes. No layer >14 or future token/readout/outcome-derived input features.',
        'Primary checkpoint: 32 already-generated tokens, at layer 14 predicting token 33.',
        'Checkpoint 0 predicts token 1 from the prompt only. Later checkpoints condition on still-active generation.',
        'Final strict correctness is the supervised target; parse failures are included, not excluded using future eligibility.',
        'Scores are not calibrated error probabilities. All preprocessing, epochs, hyperparameters and thresholds are selected without test tuning.', '',
        '| Prior tokens | Features | Model | Val AUROC | Test AUROC | AP | Recall | Precision | Test coverage |',
        '|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f'| {r["checkpoint"]} | {r["group"]} | {r["model"]} | {r["validation_auroc"]:.3f} | {r["auroc"]:.3f} | {r["pr_auc_ap"]:.3f} | {r["recall"]:.3f} | {r["precision"]:.3f} | {r["test_count"]}/1319 |')
    lines += ['', '## Primary Checkpoint', '']
    for r in results:
        if r['checkpoint'] == 32:
            lines.append(f'- {r["group"]}/{r["model"]}: AUROC 95% CI {r["auroc_ci95"]}.')
    lines += ['', 'Multilayer minus layer-14-only paired bootstrap (no multiplicity correction):', json.dumps(comparisons, indent=2), '',
        'This validates risk prediction, not an accuracy-improving model intervention. Prefixes were generated by the original full-depth Qwen.',
        'A deployment would require an instrumented forward pass and measured readout/probe overhead. Runtime improvement and calibration are not established.',
        'Use a separate intervention evaluation: no trigger versus random trigger versus classifier-triggered retry/checking, under equal budgets.',
        'Original test aggregates and earlier classifiers were inspected; use new held-out data for a confirmatory claim.']
    (output / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    base.write_json(output / 'complete.json', {'completed_unix': time.time(),
        'files': {p.name: base.sha(p) for p in output.iterdir() if p.is_file() and p.name != 'complete.json'}})


if __name__ == '__main__':
    main()
