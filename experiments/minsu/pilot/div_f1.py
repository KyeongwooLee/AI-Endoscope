"""Experiment-1 divergent-token probability, scored with the same solution-level F1 (position given)."""
import json, collections, numpy as np
from sklearn.model_selection import StratifiedGroupKFold
meta = json.load(open('feat/meta.json')); P = len(meta)
y = np.array([m['type'] for m in meta]); q = np.array([m['q'] for m in meta])
st = np.array([-m['lp_trans_tok'] for m in meta]); so = np.array([-m['lp_orig_tok'] for m in meta])
f1 = lambda a, b: 2 * a * b / (a + b) if a + b else 0.0
res_t, res_o = np.zeros(P, bool), np.zeros(P, bool)
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(P), y, q):
    grid = np.unique(np.quantile(np.r_[st[tr], so[tr]], np.linspace(0, 1, 401)))
    tau = max(grid, key=lambda t: f1(np.mean(st[tr] > t), np.mean(so[tr] <= t)))
    res_t[te], res_o[te] = st[te] > tau, so[te] <= tau
print(f'divergent-token prob (position given): acc_err={res_t.mean():.3f} acc_cor={res_o.mean():.3f} F1={f1(res_t.mean(), res_o.mean()):.3f}')
