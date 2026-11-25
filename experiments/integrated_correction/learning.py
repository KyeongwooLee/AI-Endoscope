"""Grouped probe fitting, logistic calibration, and prefix success heads."""

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

from .answers import CLASSES
from .model import HIDDEN_SIZE

C_GRID = (.01, .1, 1., 10.)


def calibration(logits, targets, weights=None):
    scores = np.asarray(logits).reshape(-1)
    targets = np.asarray(targets)
    weights = np.ones(len(targets)) if weights is None else np.asarray(weights)
    if not len(targets) or not (np.any(targets > 0) and np.any(targets < 1)):
        return {"slope": 1., "intercept": 0., "status": "insufficient_calibration_outcomes"}
    x = np.tile(scores, 2).reshape(-1, 1)
    y = np.repeat([1, 0], len(targets))
    w = np.r_[targets * weights, (1 - targets) * weights]
    clf = LogisticRegression(C=1e6).fit(x[w > 0], y[w > 0], sample_weight=w[w > 0])
    return {"slope": float(clf.coef_[0, 0]), "intercept": float(clf.intercept_[0]), "status": "fitted"}


def binary_metrics(targets, probabilities):
    y, p = np.asarray(targets), np.asarray(probabilities, dtype=float)
    positive = int(y.sum())
    negative = len(y) - positive
    return {"positive": positive, "negative": negative,
            "auroc": float(roc_auc_score(y, p)) if positive and negative else None,
            "auprc": float(average_precision_score(y, p)) if positive and negative else None,
            "f1": float(f1_score(y, p >= .5, zero_division=0)) if len(y) else None,
            **probability_metrics(y, p)}


def probability_metrics(targets, probabilities, weights=None):
    y, p = np.asarray(targets, dtype=float), np.asarray(probabilities, dtype=float)
    w = np.ones(len(y)) if weights is None else np.asarray(weights)
    if not len(y):
        return {"n": 0, "brier": None, "bce": None, "ece": None, "calibration": []}
    clipped = p.clip(1e-7, 1 - 1e-7)
    reliability = []
    for low in range(10):
        select = np.minimum((p * 10).astype(int), 9) == low
        if select.any():
            reliability.append({"bin": low, "count": int(select.sum()), "weight": float(w[select].sum()),
                                "probability": float(np.average(p[select], weights=w[select])),
                                "observed": float(np.average(y[select], weights=w[select]))})
    return {"n": len(y), "brier": float(np.average((p - y)**2, weights=w)),
            "bce": float(np.average(-y * np.log(clipped) - (1 - y) * np.log1p(-clipped), weights=w)),
            "ece": sum(b["weight"] * abs(b["probability"] - b["observed"]) for b in reliability) / w.sum(),
            "calibration": reliability}


def fit_probe(fit, tune, cal, minimum=1, text=False):
    x, y = fit
    positive = int(sum(y))
    negative = len(y) - positive
    counts = {"positive": positive, "negative": negative}
    if min(positive, negative) < minimum:
        return {"disabled": f"fit requires {minimum} positive and negative examples", "fit": counts}

    if text:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, sublinear_tf=True)
        if not any(vectorizer.build_analyzer()(item) for item in x):
            return {"disabled": "empty fit vocabulary", "fit": counts}
        train = vectorizer.fit_transform(x)
        transform = vectorizer.transform
    else:
        scaler = StandardScaler().fit(x)
        train = scaler.transform(x)
        transform = scaler.transform

    tune_usable = len(set(tune[1])) == 2
    tune_x = transform(tune[0]) if tune_usable else None
    best = None
    for c in C_GRID if tune_usable else C_GRID[:1]:
        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=1000).fit(train, y)
        score = float(roc_auc_score(tune[1], clf.decision_function(tune_x))) if tune_usable else None
        if best is None or (score is not None and score > best[0]):
            best = score, c, clf

    score, c, clf = best
    logits = clf.decision_function(transform(cal[0])) if len(cal[1]) else []
    checkpoint = {"coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
                  "calibration": calibration(logits, cal[1]), "c": c, "tune_auroc": score, "fit": counts}
    if text:
        checkpoint.update(vocab={word: int(i) for word, i in vectorizer.vocabulary_.items()},
                          idf=vectorizer.idf_.tolist())
    else:
        checkpoint.update(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist())
    return checkpoint


def predict_probe(probe, value):
    if probe.get("disabled"):
        return None
    if "vocab" in probe:
        vectorizer = TfidfVectorizer(vocabulary=probe["vocab"], ngram_range=(1, 2), sublinear_tf=True)
        vectorizer.idf_ = np.asarray(probe["idf"])
        logit = float((vectorizer.transform([value]) @ np.asarray(probe["coef"])).item()) + probe["intercept"]
    else:
        x = torch.as_tensor(value, dtype=torch.float64)
        scaled = (x - torch.as_tensor(probe["mean"], dtype=x.dtype)) / torch.as_tensor(probe["scale"], dtype=x.dtype)
        logit = scaled @ torch.as_tensor(probe["coef"], dtype=x.dtype) + probe["intercept"]
    cal = probe["calibration"]
    return float(torch.sigmoid(torch.as_tensor(logit, dtype=torch.float64) * cal["slope"] + cal["intercept"]))


def learn_probes(examples):
    results = {}
    for view in ("r_mean", "r_mean_k5", "q_end", "confidence", "tfidf"):
        for target in ("incorrect", *CLASSES):
            if target == "incorrect" and view == "r_mean_k5":
                continue
            pools = {fold: ([], []) for fold in ("fit", "tune", "calibration")}
            for fold, labels, responses, features in examples:
                fit_x, fit_y = pools[fold]
                if target == "incorrect":
                    count = len(responses) if fold == "fit" and view != "q_end" else 1
                    for i in range(count):
                        fit_x.append(responses[i]["text"] if view == "tfidf" else features[view][i])
                        fit_y.append(not responses[i]["grading"]["correct"])
                else:
                    if view == "r_mean_k5" and len(responses) != 5:
                        continue
                    value = (features["r_mean"].mean(0) if view == "r_mean_k5" else
                             responses[0]["text"] if view == "tfidf" else features[view][0])
                    fit_x.append(value)
                    fit_y.append(labels[target])
            results[f"{view}/{target}"] = fit_probe(*pools.values(), minimum=1 if target == "incorrect" else 30,
                                                            text=view == "tfidf")
    primary = results["r_mean/incorrect"]
    calibration_scores = [predict_probe(primary, f["r_mean"][0])
                          for fold, _, _, f in examples if fold == "calibration"] if not primary.get("disabled") else []
    threshold = (float(np.sort(calibration_scores)[-int(np.ceil(.25 * len(calibration_scores)))])
                 if calibration_scores else None)
    return {"probes": results, "risk_threshold": threshold,
            "threshold_calibration_n": len(calibration_scores),
            "threshold_calibration_rate": (float(np.mean(np.asarray(calibration_scores) >= threshold))
                                           if calibration_scores else None)}


class RewardHead(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(HIDDEN_SIZE, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, residual):
        return super().forward(residual.float()).squeeze(-1)


def weight_by_problem(problem_ids):
    ids, counts = np.unique(problem_ids, return_counts=True)
    return 1. / counts[np.searchsorted(ids, problem_ids)]


def reward_values(head, residual, cal):
    return torch.sigmoid(head(residual) * cal["slope"] + cal["intercept"])


def train_reward(pools, minimum, device, seed):
    x, y, ids = pools["fit"]
    problems = len(set(ids))
    metadata = {"fit_problems": problems, "fit_prefixes": len(y)}
    if problems < minimum or not (np.any(y > 0) and np.any(y < 1)):
        return None, {**metadata, "disabled": "insufficient fit problems or uniform continuation outcomes"}

    tx, ty, tids = pools["tune"]
    torch.manual_seed(seed)
    head = RewardHead().to(device)
    optim = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    weights = weight_by_problem(ids)
    weights /= weights.mean()
    tune_weights = weight_by_problem(tids)
    random = np.random.default_rng(seed)
    best, best_loss, stale, best_epoch = None, float("inf"), 0, None
    for epoch in range(1, 31):
        for indices in np.array_split(random.permutation(len(y)), (len(y) + 255) // 256):
            xb = torch.as_tensor(x[indices], dtype=torch.float32, device=device)
            yb = torch.as_tensor(y[indices], dtype=torch.float32, device=device)
            wb = torch.as_tensor(weights[indices], dtype=torch.float32, device=device)
            optim.zero_grad()
            (F.binary_cross_entropy_with_logits(head(xb), yb, reduction="none") * wb).mean().backward()
            optim.step()
        # A two-problem smoke run may have no tune fold.
        vx, vy, vw = (tx, ty, tune_weights) if len(ty) else (x, y, weights)
        logits = score_reward(head, vx)
        loss = float(np.average(np.logaddexp(0, logits) - vy * logits, weights=vw))
        if loss < best_loss:
            best = {name: tensor.detach().cpu().clone() for name, tensor in head.state_dict().items()}
            best_loss, stale, best_epoch = loss, 0, epoch
        else:
            stale += 1
            if stale == 3:
                break
    head.load_state_dict(best)
    cx, cy, cids = pools["calibration"]
    cal_logits = score_reward(head, cx)
    cal_weights = weight_by_problem(cids)
    cal = calibration(cal_logits, cy, cal_weights)
    return best, {**metadata, "epoch": best_epoch, "tune_bce": best_loss if len(ty) else None,
                  "calibration": cal, "calibration_metrics": probability_metrics(
                      cy, torch.sigmoid(torch.from_numpy(cal_logits) * cal["slope"] + cal["intercept"]).numpy(),
                      cal_weights)}


@torch.no_grad()
def score_reward(head, vectors):
    device = next(head.parameters()).device
    return np.concatenate([head(torch.as_tensor(vectors[i:i + 1024], dtype=torch.float32, device=device)).cpu().numpy()
                           for i in range(0, len(vectors), 1024)]) if len(vectors) else np.empty(0)


def load_rewards(weights_path, metadata, device):
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    heads = {}
    for name, state in weights.items():
        head = RewardHead().to(device)
        head.load_state_dict(state)
        heads[name] = (head.eval().requires_grad_(False), metadata[name]["calibration"])
    return heads
