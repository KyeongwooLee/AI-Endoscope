import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path('outputs/midlayer-classifier-v1')
results = json.loads((root / 'results.json').read_text())['results']
populations = json.loads((root / 'populations.json').read_text())
checkpoints = [0, 32, 64, 128]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True, gridspec_kw={'width_ratios':[2, 1]})
styles = [('mid14', 'logistic', '#567080', '-'), ('mid14', 'mlp', '#567080', '--'),
          ('mid_multi', 'logistic', '#a84e69', '-'), ('mid_multi', 'mlp', '#a84e69', '--')]
for group, model, color, style in styles:
    rows = [next(r for r in results if r['checkpoint'] == t and r['group'] == group and r['model'] == model) for t in checkpoints]
    axes[0].plot(checkpoints, [r['auroc'] for r in rows], marker='o', color=color, linestyle=style,
                 label=('Layer 14 only' if group == 'mid14' else 'Layers <=14') + ' / ' + model)
    primary = rows[1]
    low, high = primary['auroc_ci95']
    axes[0].errorbar(32, primary['auroc'], yerr=[[primary['auroc']-low],[high-primary['auroc']]],
                     color=color, capsize=3, alpha=.65)
axes[0].axhline(.5, linestyle=':', color='#555555')
axes[0].set(xticks=checkpoints, ylim=(.45,.7), xlabel='Previously generated tokens', ylabel='Test AUROC',
            title='Depth-limited prefix probes (primary checkpoint: 32)')
axes[0].legend(frameon=False, fontsize=8)
counts = [populations[str(t)]['test_count'] for t in checkpoints]
axes[1].bar(range(4), np.array(counts)/1319, color='#247b83', width=.6)
for i, count in enumerate(counts):
    axes[1].text(i, count/1319+.025, str(count), ha='center', fontsize=9)
axes[1].set(xticks=range(4), xticklabels=checkpoints, ylim=(0,1.12), xlabel='Previously generated tokens',
            ylabel='Fraction of 1,319 test problems', title='Still-active generation coverage')
for axis in axes:
    axis.spines[['top','right']].set_visible(False)
    axis.grid(axis='y', alpha=.15)
fig.savefig('outputs/midlayer-classifier-summary.png', dpi=160)
