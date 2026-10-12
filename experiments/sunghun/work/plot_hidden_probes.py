import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1] / 'outputs/hidden-probes-v1/results'
rows = json.loads((root / 'results.json').read_text())
variants = ('h14', 'h14_delta13', 'h7_h10_h14')
labels = ('h14', 'h14 + (h14 - h13)', 'h7 + h10 + h14')
fig, axis = plt.subplots(figsize=(10, 5), layout='constrained')
for offset, kind, color in ((-.15, 'linear', '#167a68'), (.15, 'mlp', '#c44852')):
    subset = [next(row for row in rows if row['checkpoint'] == 32 and row['variant'] == variant and row['model'] == kind) for variant in variants]
    mean = np.array([row['auroc'] for row in subset])
    ci = np.array([row['auroc_ci95'] for row in subset]).T
    axis.errorbar(np.arange(3) + offset, mean, yerr=np.array([mean - ci[0], ci[1] - mean]),
        fmt='o', markersize=8, capsize=4, color=color, label=kind.upper())
    for index, value in enumerate(mean):
        axis.text(index + offset, ci[1, index] + .01, f'{value:.3f}',
            ha='center', fontsize=10, color=color)
axis.axhline(.5, linestyle='--', linewidth=1, color='#777777', label='Chance AUROC')
axis.set_xticks(np.arange(3), labels)
axis.set_ylim(.48, .72)
axis.set_ylabel('Test AUROC, incorrect = positive')
axis.set_title('Raw hidden-state probes at decoder block 14, after 32 generated tokens')
axis.grid(axis='y', alpha=.2)
axis.legend(loc='upper left')
fig.savefig(root / 'hidden_probe_summary.png', dpi=160)
