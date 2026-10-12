import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

workspace = Path(__file__).resolve().parents[1]
root = workspace / 'outputs/feature-ablation-v1'
frame = pd.read_parquet(root / 'input/engineered_features.parquet')
with np.load(root / 'input/designs.npz', allow_pickle=False) as data:
    ids, y, train, valid = [data[key] for key in ('ids', 'y', 'train', 'valid')]
frame = frame.loc[ids]
spec = importlib.util.spec_from_file_location('hp', workspace / 'outputs/gsm8k-logit-study/scripts/train_hidden_probes.py')
hp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hp)
names = ['current_stability_L10_entropy_prefix_mean', 'current_stability_L7_entropy_prefix_mean',
         'change_stability_L7_entropy_prefix_mean']
scan = pd.read_csv(root / 'results/univariate_scan.csv').set_index('feature')
intervals = {}
for name in names:
    sign = int(scan.loc[name, 'direction_from_train'])
    intervals[name] = {'validation_auroc': float(scan.loc[name, 'validation_auroc']),
        'ci95': hp.bootstrap_auc(y[valid], sign * frame[name].to_numpy()[valid], replicates=1000),
        'direction_from_train': sign, 'warning': 'Exploratory pointwise CI after a many-feature scan.'}
(root / 'results/candidate_intervals.json').write_text(json.dumps(intervals, indent=2))
fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
for column, (name, title) in enumerate(zip(names[:2], ['Layer 10 prefix-mean entropy', 'Layer 7 prefix-mean entropy'])):
    values = frame[name].to_numpy()
    bins = np.linspace(values.min(), values.max(), 17)
    for row, (mask, partition) in enumerate([(train, 'Train'), (valid, 'Validation')]):
        axis = axes[row, column]
        for target, label, color in [(0, 'Correct', '#167a68'), (1, 'Incorrect', '#c44852')]:
            selected = values[mask & (y == target)]
            axis.hist(selected, bins=bins, density=True, histtype='step', color=color, linewidth=2,
                      label=f'{label} (n={len(selected)})')
            axis.axvline(selected.mean(), color=color, linestyle=':', linewidth=1)
        axis.set_title(f'{partition}: {title}', fontsize=11)
        axis.set_xlabel('Entropy (nats), positions 0..32')
        axis.set_ylabel('Within-class density')
        axis.legend(fontsize=8)
        axis.grid(axis='y', alpha=.2)
fig.suptitle('Train-selected entropy candidates: lower intermediate entropy in incorrect runs', fontsize=13)
fig.savefig(root / 'results/entropy_candidates.png', dpi=160)
