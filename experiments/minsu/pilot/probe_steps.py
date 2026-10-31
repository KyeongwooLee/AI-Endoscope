"""Pilot 2: find the first wrong step WITHOUT being told where it is (ProcessBench-style).

Each step gets a suspicion score. Scanning a solution in order, the first step whose score exceeds a
threshold (chosen on the training folds) is the predicted first error; no crossing = "no error".
  acc_err : erroneous (transformed) solutions where the predicted first error == true wrong step
  acc_cor : correct (original) solutions with no step flagged
  F1      : harmonic mean of the two
"""
import json, sys, collections, warnings
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
F = sys.argv[1]
meta = json.load(open(f'{F}/steps_meta.json'))
S = len(meta)
pid = np.array([m['pid'] for m in meta]); lab = np.array([m['label'] for m in meta])
is_trans = np.array([m['sol'] == 'trans' for m in meta]); kk = np.array([m['k'] for m in meta])
P = pid.max() + 1
ptype = np.empty(P, dtype=object); pq = np.empty(P, dtype=object); perr = np.zeros(P, int)
for m in meta:
    ptype[m['pid']], pq[m['pid']], perr[m['pid']] = m['type'], m['q'], m['err']
types = sorted(set(ptype))
short = dict(adding_irrelevant_information='HA', calculation_error='CA', confusing_formula_error='FC', counting_error='CO',
             missing_step='MS', operator_error='OP', referencing_context_value_error='CV',
             referencing_previous_step_value_error='PS', unit_conversion_error='UC')
pfold = np.zeros(P, int)
for f, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(P), ptype.astype(str), pq)):
    pfold[te] = f
sfold = pfold[pid]
print(f'pairs={P} steps={S}  labels: correct={np.sum(lab == 0)} first-wrong={np.sum(lab == 1)} after-error={np.sum(lab == 2)}')

# per-solution step index lists, in step order
sol_steps = collections.defaultdict(list)
for i in np.lexsort((kk, is_trans, pid)):
    sol_steps[(pid[i], bool(is_trans[i]))].append(i)


def judge(score, pairs, tau):
    """returns per-pair outcome for transformed ('exact'/'early'/'late_or_miss') and bool for original (no alarm)."""
    out_t, out_o = {}, {}
    for p in pairs:
        flagged = [kk[i] for i in sol_steps[(p, True)] if score[i] > tau]
        first = flagged[0] if flagged else None
        out_t[p] = 'exact' if first == perr[p] else ('early' if first is not None and first < perr[p] else 'late_or_miss')
        out_o[p] = not any(score[i] > tau for i in sol_steps[(p, False)])
    return out_t, out_o


idx_of = {(pid[i], bool(is_trans[i]), kk[i]): i for i in range(S)}


def judge_oracle(score, pairs, tau):
    """position given: judge only the true first-wrong step (transformed) and the same step of the original."""
    out_t = {p: 'exact' if score[idx_of[(p, True, perr[p])]] > tau else 'late_or_miss' for p in pairs}
    out_o = {p: bool(score[idx_of[(p, False, perr[p])]] <= tau) for p in pairs}
    return out_t, out_o


def pick_tau_oracle(score, pairs):
    sub = np.array([idx_of[(p, t, perr[p])] for p in pairs for t in (True, False)])
    grid = np.unique(np.quantile(score[sub], np.linspace(0, 1, 201)))
    return max(grid, key=lambda t: f1_of(*judge_oracle(score, pairs, t))[2])


def f1_of(out_t, out_o):
    a = np.mean([v == 'exact' for v in out_t.values()]); b = np.mean(list(out_o.values()))
    return a, b, (2 * a * b / (a + b) if a + b else 0.0)


def pick_tau(score, pairs):
    sub = np.concatenate([sol_steps[(p, t)] for p in pairs for t in (True, False)])
    grid = np.unique(np.quantile(score[sub], np.linspace(0.3, 1, 150)))
    return max(grid, key=lambda t: f1_of(*judge(score, pairs, t))[2])


def evaluate(name, score_fn):
    oof = np.zeros(S); out_t, out_o = {}, {}; or_t, or_o = {}, {}
    for f in range(5):
        tr_p, te_p = np.where(pfold != f)[0], np.where(pfold == f)[0]
        s = score_fn(f)                       # scores for ALL steps from a model fit on fold != f
        tau = pick_tau(s, tr_p)
        oof[sfold == f] = s[sfold == f]
        t_, o_ = judge(s, te_p, tau); out_t.update(t_); out_o.update(o_)
        t_, o_ = judge_oracle(s, te_p, pick_tau_oracle(s, tr_p)); or_t.update(t_); or_o.update(o_)
    m01 = lab != 2
    auc = roc_auc_score(lab[m01], oof[m01])
    a, b, f1 = f1_of(out_t, out_o)
    br = collections.Counter(out_t.values())
    per = ' '.join(f"{short[t]}={np.mean([out_t[p] == 'exact' for p in range(P) if ptype[p] == t]):.2f}" for t in types)
    return name, auc, a, b, f1, {k: br[k] / P for k in ['exact', 'early', 'late_or_miss']}, per, (out_t, out_o), f1_of(or_t, or_o)


def baseline(key):
    v = np.array([m[key] for m in meta])
    return lambda f: v


def hidden(feat, layer):
    def fn(f):
        X = np.load(f'{F}/X_{feat}.npy', mmap_mode='r')[:, layer].astype(np.float32)
        tr = (sfold != f) & (lab != 2)
        m = make_pipeline(StandardScaler(), PCA(256, random_state=0),
                          LogisticRegression(C=0.05, class_weight='balanced', max_iter=2000)).fit(X[tr], lab[tr])
        return m.decision_function(X)
    return fn


def hidden_plus_nll(feat, layer):
    """hidden (PCA) + the two confidence scalars: does the hidden state add anything beyond confidence?"""
    nll = np.array([[m['nll_max'], m['nll_mean']] for m in meta], dtype=np.float32)
    def fn(f):
        X = np.load(f'{F}/X_{feat}.npy', mmap_mode='r')[:, layer].astype(np.float32)
        tr = (sfold != f) & (lab != 2)
        red = make_pipeline(StandardScaler(), PCA(256, random_state=0)).fit(X[tr])
        Z = np.hstack([red.transform(X), StandardScaler().fit(nll[tr]).transform(nll)])
        m = LogisticRegression(C=0.05, class_weight='balanced', max_iter=2000).fit(Z[tr], lab[tr])
        return m.decision_function(Z)
    return fn


def nll_only_lr():
    """the two confidence scalars combined by the same logistic regression (fair counterpart of hidden+NLL)"""
    nll = np.array([[m['nll_max'], m['nll_mean']] for m in meta], dtype=np.float32)
    def fn(f):
        tr = (sfold != f) & (lab != 2)
        m = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=2000)).fit(nll[tr], lab[tr])
        return m.decision_function(nll)
    return fn


configs = [(f'baseline: step NLL max (lowest token prob)', baseline('nll_max')),
           (f'baseline: step NLL mean', baseline('nll_mean'))]
for feat in ['end', 'mean']:
    for l in [4, 8, 12, 16, 20, 24, 28]:
        configs.append((f'hidden {feat:4s} L{l}', hidden(feat, l)))
configs += [('NLL max+mean (LR)', nll_only_lr()), ('hidden mean L16 + NLL', hidden_plus_nll('mean', 16)),
            ('hidden mean L28 + NLL', hidden_plus_nll('mean', 28))]
with parallel_backend('loky', inner_max_num_threads=8):
    results = Parallel(n_jobs=len(configs))(delayed(evaluate)(n, fn) for n, fn in configs)

print('\n## first-wrong-step localization, no oracle position (question-grouped 5-fold)')
print('   step AUROC: all correct steps vs first-wrong steps across whole solutions')
print(f"  {'method':38s} {'stepAUROC':>9s} {'acc_err':>8s} {'acc_cor':>8s} {'F1':>6s} | erroneous sols: exact / flagged-too-early / late-or-miss")
for name, auc, a, b, f1, br, per, _, _o in results:
    print(f"  {name:38s} {auc:9.3f} {a:8.3f} {b:8.3f} {f1:6.3f} | {br['exact']:.2f} / {br['early']:.2f} / {br['late_or_miss']:.2f}")
best = max([r for r in results if r[0].startswith('hidden mean') and '+' not in r[0]], key=lambda r: r[4])
print('\n  acc_err per type')
R = {r[0]: r for r in results}
for r in [results[0], best, R['hidden mean L28 + NLL']]:
    print(f'   {r[0]:38s} {r[6]}')

# paired bootstrap over question groups for F1 differences
rng = np.random.default_rng(0)
uq = sorted(set(pq)); qidx = collections.defaultdict(list)
for p in range(P):
    qidx[pq[p]].append(p)


def f1_from(out, pairs):
    a = np.mean([out[0][p] == 'exact' for p in pairs]); b = np.mean([out[1][p] for p in pairs])
    return 2 * a * b / (a + b) if a + b else 0.0


print('\n## paired bootstrap (resampling questions, 2000x): F1(A) - F1(B), 95% CI')
for A_, B_ in [(best[0], results[0][0]), ('hidden mean L28 + NLL', results[0][0]), ('hidden mean L28 + NLL', 'NLL max+mean (LR)')]:
    d = []
    for _ in range(2000):
        pairs = [p for qq in rng.choice(uq, len(uq)) for p in qidx[qq]]
        d.append(f1_from(R[A_][7], pairs) - f1_from(R[B_][7], pairs))
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f'  {A_:24s} vs {B_:42s}: {np.mean(d):+.3f}  [{lo:+.3f}, {hi:+.3f}]')

print('\n## SAME metric, two conditions: position given (oracle) vs not given (scan all steps)')
print('   acc_err = erroneous solutions judged right, acc_cor = correct solutions judged right, F1 = harmonic mean')
print(f"  {'method':38s} | {'given: acc_err':>14s} {'acc_cor':>8s} {'F1':>6s} | {'not given: acc_err':>18s} {'acc_cor':>8s} {'F1':>6s}")
for r in results:
    (ao, bo, fo), a, b, f1 = r[8], r[2], r[3], r[4]
    print(f"  {r[0]:38s} | {ao:14.3f} {bo:8.3f} {fo:6.3f} | {a:18.3f} {b:8.3f} {f1:6.3f}")
