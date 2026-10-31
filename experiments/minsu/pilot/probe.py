import json, sys, collections, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

warnings.filterwarnings('ignore')
F = sys.argv[1]
meta = json.load(open(f'{F}/meta.json'))
A = {k: np.load(f'{F}/{k}.npy').astype(np.float32) for k in ['o_div', 't_div', 'o_end', 't_end']}
N, L1, D = A['t_end'].shape
y = np.array([m['type'] for m in meta]); q = np.array([m['q'] for m in meta]); ds = np.array([m['ds'] for m in meta])
types = sorted(set(y))
short = dict(adding_irrelevant_information='HA', calculation_error='CA', confusing_formula_error='FC', counting_error='CO',
             missing_step='MS', operator_error='OP', referencing_context_value_error='CV',
             referencing_previous_step_value_error='PS', unit_conversion_error='UC')
print(f'N={N} layers={L1} dim={D}  by ds={dict(collections.Counter(ds))}')

cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
folds = list(cv.split(np.zeros(N), y, q))


def probe(n_out):
    if n_out == 2:
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
    return make_pipeline(StandardScaler(), PCA(128, random_state=0), LogisticRegression(C=0.1, max_iter=3000))


def paired_binary(X1, X0, groups_split=folds):
    """wrong (X1) vs correct (X0) of the same pair; both members of a pair stay in the same fold."""
    s = np.zeros(2 * N)
    for tr, te in groups_split:
        Xtr = np.concatenate([X1[tr], X0[tr]]); ytr = np.r_[np.ones(len(tr)), np.zeros(len(tr))]
        m = probe(2).fit(Xtr, ytr)
        s[te] = m.predict_proba(X1[te])[:, 1]; s[N + te] = m.predict_proba(X0[te])[:, 1]
    return s


yb = np.r_[np.ones(N), np.zeros(N)]
yt2 = np.r_[y, y]
print('\n## [A] correct vs wrong step (paired, question-grouped 5-fold), AUROC')
for name, a, b in [('step NLL mean', 't_step_nll_mean', 'o_step_nll_mean'), ('step NLL max', 't_step_nll_max', 'o_step_nll_max')]:
    s = np.r_[[m[a] for m in meta], [m[b] for m in meta]]
    print(f'  baseline {name:14s}: {roc_auc_score(yb, s):.3f}  per type: ' +
          ' '.join(f'{short[t]}={roc_auc_score(yb[yt2 == t], s[yt2 == t]):.2f}' for t in types))
layers = list(range(0, L1, 2)) + ([L1 - 1] if (L1 - 1) % 2 else [])
best = (0, None, None)
for pos in ['end', 'div']:
    line = []
    for l in layers:
        s = paired_binary(A[f't_{pos}'][:, l], A[f'o_{pos}'][:, l])
        auc = roc_auc_score(yb, s); line.append(f'L{l}:{auc:.2f}')
        if pos == 'end' and auc > best[0]:
            best = (auc, l, s)
    print(f'  hidden @{pos}: ' + ' '.join(line))
auc, bl, s = best
print(f'  best @end layer {bl}: {auc:.3f}  per type: ' + ' '.join(f'{short[t]}={roc_auc_score(yb[yt2 == t], s[yt2 == t]):.2f}' for t in types))

print('\n## [B] 9-way error type, macro-F1 (chance 0.11), question-grouped 5-fold')
multi = np.array([len(set(y[q == qq])) > 1 for qq in q])
print(f'  samples whose question appears with >=2 types: {multi.sum()}')
res = {}
for name, X in [('trans @end', A['t_end']), ('ORIG @end (topic/position control)', A['o_end']),
                ('trans-orig @end (diff, analysis only)', A['t_end'] - A['o_end']), ('trans @div', A['t_div'])]:
    line = []
    for l in [0, 4, 8, 12, 16, 20, 24, L1 - 1]:
        p = cross_val_predict(probe(9), X[:, l], y, cv=folds)
        f = f1_score(y, p, average='macro'); res[(name, l)] = p
        line.append(f'L{l}:{f:.2f}')
    print(f'  {name:40s} ' + ' '.join(line))
bl9 = max([0, 4, 8, 12, 16, 20, 24, L1 - 1], key=lambda l: f1_score(y, res[('trans @end', l)], average='macro'))
for name in ['trans @end', 'ORIG @end (topic/position control)']:
    p = res[(name, bl9)]
    print(f'  L{bl9} {name:38s} all={f1_score(y, p, average="macro"):.3f}  multi-type-question subset={f1_score(y[multi], p[multi], average="macro"):.3f}')
p = res[('trans @end', bl9)]
print(f'\n  confusion (trans @end, L{bl9}); rows=true, cols=pred, order=' + ' '.join(short[t] for t in types))
for t, row in zip(types, confusion_matrix(y, p, labels=types)):
    print(f'   {short[t]:5s}' + ''.join(f'{v:5d}' for v in row))

print('\n## [B2] GSM8K -> MathQA transfer (trans @end)')
for l in [bl9]:
    tr, te = ds == 'GSM8K', ds == 'MathQA'
    m = probe(9).fit(A['t_end'][tr, l], y[tr]); p = m.predict(A['t_end'][te, l])
    m2 = probe(9).fit(A['o_end'][tr, l], y[tr]); p2 = m2.predict(A['o_end'][te, l])
    print(f'  L{l}: trans macro-F1={f1_score(y[te], p, average="macro"):.3f} | orig-control={f1_score(y[te], p2, average="macro"):.3f}')

print('\n## [C] would the model itself write this error? (at the divergence point)')
for t in types:
    sub = [m for m in meta if m['type'] == t]
    lt = np.array([m['lp_trans_tok'] for m in sub]); lo = np.array([m['lp_orig_tok'] for m in sub])
    print(f'  {short[t]:5s} P(err tok) median={np.exp(np.median(lt)):.3f}  err tok more likely than orig tok: {np.mean(lt > lo):.0%}')

print('\n## [D] memorization indicator: teacher-forced top-1 agreement on ORIGINAL solution')
import re
norm = lambda s: re.sub(r'\s+', ' ', s).strip()
split = {}
for sp in ['train', 'test']:
    for line in open(f'{sys.argv[2]}/{sp}.jsonl'):
        split[norm(json.loads(line)['question'])] = sp
groups = collections.defaultdict(list)
for m in meta:
    key = m['ds'] if m['ds'] == 'MathQA' else 'GSM8K-' + split.get(norm(m['q']), 'unknown')
    groups[key].append(m['orig_top1_acc'])
for k, v in sorted(groups.items()):
    v = np.array(v); print(f'  {k:14s} n={len(v):4d}  mean {v.mean():.3f}  (>0.95: {np.mean(v > 0.95):.0%})')
