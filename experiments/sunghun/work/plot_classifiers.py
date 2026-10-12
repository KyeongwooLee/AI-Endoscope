import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path('outputs/error-classifier-v1')
data = json.loads((root / 'results.json').read_text())
groups = ['length', 'confidence', 'internal', 'combined', 'confidence_length', 'combined_length']
labels = ['Length', 'Confidence', 'Internal', 'Conf. + internal', 'Conf. + length', 'All + length']
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
for axis, metric, title, baseline in zip(axes, ['auroc', 'pr_auc_ap'],
                                        ['Test AUROC', 'Test Average Precision'],
                                        [.5, data['test_error_prevalence']]):
    x = np.arange(len(groups))
    for shift, model, color, label in [(-.18, 'logistic', '#247b83', 'Logistic regression'),
                                        (.18, 'mlp', '#b45570', 'MLP (3-seed mean)')]:
        values = [next(r[metric] for r in data['results'] if r['group'] == g and r['model'] == model) for g in groups]
        axis.bar(x + shift, values, .34, color=color, label=label)
    axis.axhline(baseline, linestyle='--', color='#555555', linewidth=1)
    axis.set_xticks(x, labels, rotation=25, ha='right')
    axis.set_ylim(.3 if metric == 'auroc' else 0, .9 if metric == 'auroc' else .8)
    axis.set_title(title)
    axis.spines[['top', 'right']].set_visible(False)
    axis.grid(axis='y', alpha=.15)
axes[0].legend(loc='upper left', frameon=False, fontsize=9)
fig.suptitle('GSM8K error classification: 790 train / 198 validation / 1,294 eligible test')
fig.savefig('outputs/error-classifier-summary.png', dpi=160)
print('Saved classifier summary figure')
