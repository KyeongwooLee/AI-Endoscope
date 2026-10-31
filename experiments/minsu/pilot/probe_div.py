import json, numpy as np, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, f1_score
warnings.filterwarnings("ignore")
meta = json.load(open("feat/meta.json"))
y = np.array([m["type"] for m in meta]); q = np.array([m["q"] for m in meta]); N = len(y)
types = sorted(set(y))
short = dict(adding_irrelevant_information="HA", calculation_error="CA", confusing_formula_error="FC", counting_error="CO", missing_step="MS", operator_error="OP", referencing_context_value_error="CV", referencing_previous_step_value_error="PS", unit_conversion_error="UC")
print("## [A-div] divergent-token surprisal baseline (same token as hidden @div), AUROC")
s = np.r_[[-m["lp_trans_tok"] for m in meta], [-m["lp_orig_tok"] for m in meta]]
yb = np.r_[np.ones(N), np.zeros(N)]; yt2 = np.r_[y, y]
print(f"  overall {roc_auc_score(yb, s):.3f}  per type: " + " ".join(f"{short[t]}={roc_auc_score(yb[yt2==t], s[yt2==t]):.2f}" for t in types))
print("\n## [B-div] 9-way type @div, macro-F1, question-grouped 5-fold")
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(N), y, q))
multi = np.array([len(set(y[q == qq])) > 1 for qq in q])
probe = lambda: make_pipeline(StandardScaler(), PCA(128, random_state=0), LogisticRegression(C=0.1, max_iter=3000))
for k, name in [("t_div", "trans @div"), ("o_div", "ORIG @div (control: correct token, same position)")]:
    X = np.load(f"feat/{k}.npy").astype(np.float32)
    out = []
    for l in [0, 8, 16, 28]:
        p = cross_val_predict(probe(), X[:, l], y, cv=folds)
        out.append(f"L{l}:{f1_score(y, p, average='macro'):.2f}(multi {f1_score(y[multi], p[multi], average='macro'):.2f})")
    print(f"  {name:50s} " + " ".join(out))
