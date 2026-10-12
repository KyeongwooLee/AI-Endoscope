import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1] / 'outputs/hidden-probes-v1/results'
frame = pd.read_csv(root / 'layer_change_summary.csv')
fits = json.loads((root / 'layer_shape_fits.json').read_text())
fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout='constrained')
for axis, metric, title in zip(axes, ('delta_norm', 'relative_delta', 'cosine_distance'),
                             ('Absolute update L2', 'Relative update / previous norm', 'Direction change: 1 - cosine')):
    for group, label, color in (('correct', 'Correct', '#167a68'), ('incorrect', 'Incorrect', '#c44852')):
        selected = frame[(frame.phase == 'test') & (frame.checkpoint == 32) &
            (frame.layer >= 2) & (frame.metric == metric) & (frame.group == group)]
        axis.plot(selected.layer, selected['mean'], color=color, label=label, linewidth=2)
        axis.fill_between(selected.layer, selected.ci_low, selected.ci_high, color=color, alpha=.16)
    depth = np.arange(2, 29)
    scale = fits[f't32_{metric}']['inverse_scale']
    axis.plot(depth, scale / depth, color='#315e9c', linestyle=':', linewidth=1.5, label='1/x fit (exploration)')
    axis.axvline(14, color='#777777', linestyle='--', linewidth=1)
    axis.set_title(title, fontsize=11)
    axis.set_xlabel('Decoder block')
    axis.set_xticks([2, 7, 14, 21, 28])
    axis.grid(alpha=.18)
    axis.legend(fontsize=8)
fig.suptitle('Test layer changes after 32 generated tokens; embedding transition excluded', fontsize=14)
fig.savefig(root / 'layer_changes_t32_zoom.png', dpi=160)
