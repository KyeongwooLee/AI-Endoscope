"""Descriptive layer-change curves, separating scale, relative update and direction."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

METRICS = ('previous_norm', 'hidden_norm', 'delta_norm', 'relative_delta', 'cosine_distance')
DISPLAY = {'delta_norm': 'Absolute update L2', 'relative_delta': 'Relative update L2 / previous L2',
           'cosine_distance': 'Direction change (1 - cosine)'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def mean_ci(matrix, replicates=1000):
    rng = np.random.default_rng(42)
    samples = np.stack([matrix[rng.integers(0, len(matrix), len(matrix))].mean(0) for _ in range(replicates)])
    return matrix.mean(0), np.quantile(samples, [.025, .975], axis=0)


def fit_shapes(exploration_mean, test_mean):
    # Layer 1 is the embedding-to-first-block transition, reported but not fitted.
    layer = np.arange(2, 29, dtype=float)
    train, test = exploration_mean[1:], test_mean[1:]
    assert (train > 0).all()
    exponent, log_scale = np.polyfit(np.log(layer), np.log(train), 1)
    inverse_scale = float(np.dot(train, 1 / layer) / np.dot(1 / layer, 1 / layer))
    curves = {'constant': np.full(len(layer), train.mean()),
              'inverse_1_over_layer': inverse_scale / layer,
              'power': np.exp(log_scale) * layer ** exponent}
    denominator = np.square(test - test.mean()).sum()
    return {'power_exponent': float(exponent), 'power_scale': float(np.exp(log_scale)),
        'inverse_scale': inverse_scale,
        'test_mean_r_squared': {name: float(1 - np.square(test - curve).sum() / denominator)
                              for name, curve in curves.items()},
        'interpretation': 'Descriptive fit to layer means, not an independence-based statistical test.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    data, output = Path(args.data), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    completion = json.loads((data / 'complete.json').read_text())
    rows, summaries, shape_results = [], {}, {}
    cohorts = {}
    for entry in completion['batches']:
        path = data / entry['file']
        assert sha(path) == entry['sha256']
        with np.load(path, allow_pickle=False) as batch:
            for phase in ('exploration', 'test'):
                for checkpoint in (0, 32, 64, 128):
                    mask = (batch['phases'] == phase) & (batch['checkpoints'] == checkpoint)
                    if mask.any():
                        cohorts.setdefault((phase, checkpoint), []).append((batch['ids'][mask], batch['y'][mask], batch['trajectory'][mask]))
    means = {}
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, layout='constrained')
    norm_figure, norm_axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    depth = np.arange(1, 29)
    for (phase, checkpoint), batches in cohorts.items():
        ids, y, values = [np.concatenate([batch[index] for batch in batches]) for index in range(3)]
        assert len(set(ids)) == len(ids)
        for metric_index, metric in enumerate(METRICS):
            matrix = values[:, :, metric_index]
            mean, ci = mean_ci(matrix)
            means[phase, checkpoint, metric] = mean
            for layer in range(28):
                rows.append({'phase': phase, 'checkpoint': checkpoint, 'group': 'all', 'n': len(ids),
                    'layer': layer + 1, 'metric': metric, 'mean': mean[layer], 'ci_low': ci[0, layer], 'ci_high': ci[1, layer]})
            if checkpoint != 32:
                continue
            if metric in DISPLAY:
                early = matrix[:, 1:7].mean(1)
                late = matrix[:, 21:28].mean(1)
                ratio_mean, ratio_ci = mean_ci((late / early).reshape(-1, 1))
                summaries[f'{phase}_{metric}'] = {'n': len(ids),
                    'mean_problem_late_early_ratio': float(ratio_mean[0]),
                    'ratio_ci95': ratio_ci[:, 0].tolist(),
                    'spearman_of_mean_vs_layer_2_28': float(spearmanr(depth[1:], mean[1:]).statistic),
                    'fraction_decreasing_steps_of_mean_2_28': float((np.diff(mean[1:]) < 0).mean())}
                axis = axes[0 if phase == 'exploration' else 1, list(DISPLAY).index(metric)]
                axis.plot(depth, mean, color='#555555', linewidth=1.2, linestyle=':', label='All')
                for target, label, color in ((0, 'Correct', '#167a68'), (1, 'Incorrect', '#c44852')):
                    group_mean, group_ci = mean_ci(matrix[y == target])
                    axis.plot(depth, group_mean, color=color, linewidth=1.7, label=label)
                    axis.fill_between(depth, group_ci[0], group_ci[1], color=color, alpha=.16)
                    for layer in range(28):
                        rows.append({'phase': phase, 'checkpoint': checkpoint, 'group': label.lower(),
                            'n': int((y == target).sum()), 'layer': layer + 1, 'metric': metric,
                            'mean': group_mean[layer], 'ci_low': group_ci[0, layer], 'ci_high': group_ci[1, layer]})
                axis.axvline(14, color='#777777', alpha=.7, linestyle='--', linewidth=1)
                axis.set_title(f'{phase.title()}: {DISPLAY[metric]}', fontsize=11)
                axis.set_xlabel('Decoder block (raw residual, before final norm)')
                axis.set_xticks([1, 7, 14, 21, 28])
                axis.grid(alpha=.18)
                axis.legend(fontsize=8)
                # Pointwise CIs for wrong-minus-correct, retaining per-problem sampling.
                rng = np.random.default_rng(43)
                wrong, correct = matrix[y == 1], matrix[y == 0]
                difference = wrong.mean(0) - correct.mean(0)
                boots = np.stack([wrong[rng.integers(0, len(wrong), len(wrong))].mean(0)
                    - correct[rng.integers(0, len(correct), len(correct))].mean(0) for _ in range(1000)])
                lower, upper = np.quantile(boots, [.025, .975], axis=0)
                summaries[f'{phase}_{metric}']['incorrect_minus_correct'] = {
                    'mean_by_layer': difference.tolist(), 'ci_low_by_layer': lower.tolist(),
                    'ci_high_by_layer': upper.tolist(), 'ci_type': 'Pointwise, no multiple-comparison correction.'}
            if metric == 'hidden_norm':
                axis = norm_axes[0 if phase == 'exploration' else 1]
                axis.plot(depth, mean, color='#315e9c')
                axis.fill_between(depth, ci[0], ci[1], color='#315e9c', alpha=.2)
                axis.set_title(f'{phase.title()}: residual L2 norm at token checkpoint 32')
                axis.set_xlabel('Decoder block')
                axis.set_xticks([1, 7, 14, 21, 28])
                axis.grid(alpha=.18)
    for metric in DISPLAY:
        for checkpoint in (0, 32, 64, 128):
            shape_results[f't{checkpoint}_{metric}'] = fit_shapes(
                means['exploration', checkpoint, metric], means['test', checkpoint, metric])
    figure.suptitle('Layer-to-layer residual changes at the same causal prediction position (t=32)', fontsize=14)
    figure.savefig(output / 'layer_changes_t32.png', dpi=160)
    norm_figure.savefig(output / 'residual_norms_t32.png', dpi=160)
    plt.close('all')
    pd.DataFrame(rows).to_csv(output / 'layer_change_summary.csv', index=False)
    write_json(output / 'layer_change_statistics.json', summaries)
    write_json(output / 'layer_shape_fits.json', shape_results)
    report = ['# Layer Change Analysis', '',
        'h_l is the raw output of decoder block l; h_0 is the token embedding. No final RMSNorm is mixed in.',
        'All quantities compare the SAME prediction position across layers.', '',
        '- Absolute change: ||h_l - h_(l-1)||_2.',
        '- Relative change: absolute change / ||h_(l-1)||_2.',
        '- Direction change: 1 - cosine(h_l, h_(l-1)).', '',
        'Primary checkpoint is t=32. Early window: layers 2..7; late window: layers 22..28.',
        'Embedding-to-layer-1 transition is plotted but excluded from trend fitting.', '',
        '| Quantity | Test late/early ratio (95% CI) | Power exponent fitted on exploration | Test R2 of fixed 1/x fit |',
        '|---|---|---:|---:|']
    for metric in DISPLAY:
        summary = summaries[f'test_{metric}']
        fitted = shape_results[f't32_{metric}']
        lo, hi = summary['ratio_ci95']
        report.append(f'| {metric} | {summary["mean_problem_late_early_ratio"]:.3f} [{lo:.3f}, {hi:.3f}] | '
            f'{fitted["power_exponent"]:.3f} | {fitted["test_mean_r_squared"]["inverse_1_over_layer"]:.3f} |')
    report += ['', '## Limits', '',
        'Decreasing relative changes can reflect growing residual norms, even if absolute updates do not shrink.',
        'A decreasing average trajectory is not sufficient for error detection: correct and incorrect runs must differ predictively.',
        'The trend is not assumed to follow 1/x. Power fits, constant and fixed 1/x fits are descriptive comparisons.',
        'Bootstrap unit is a problem; pointwise intervals are not simultaneous across layers or metrics.',
        'The same old test set has been inspected previously; this is exploratory, not an untouched confirmatory test.',
        'Only fixed token checkpoints were extracted, not all tokens or reasoning regions.']
    (output / 'trajectory_report.md').write_text('\n'.join(report) + '\n')
    write_json(output / 'trajectory_complete.json', {'source_sha256': sha(__file__),
        'files': {name: sha(output / name) for name in ('layer_changes_t32.png', 'residual_norms_t32.png',
            'layer_change_summary.csv', 'layer_change_statistics.json', 'layer_shape_fits.json', 'trajectory_report.md')}})
    print('TRAJECTORY_ANALYSIS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
