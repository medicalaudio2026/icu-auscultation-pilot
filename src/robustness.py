"""Adversarial robustness: white-box L2 PGD in feature space, natural-noise control,
and an adaptively evaluated mitigation (noise-augmented training + randomized
smoothing). Budget is expressed as a fraction of the feature-vector L2 norm.
"""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from .config import N_SPLITS, N_SEEDS, BASE_SEED

DEF_SIGMA = 0.5

def pgd_linear_l2(X_std, y, w, b, epsilon, steps=20, step_frac=0.25):
    X = X_std.copy().astype(float); y = np.asarray(y).astype(int)
    sign = np.where(y == 1, -1.0, 1.0); step = step_frac * epsilon
    wn = w / (np.linalg.norm(w) + 1e-12)
    for _ in range(steps):
        X = X + step * sign[:, None] * wn[None, :]
        d = X - X_std; nrm = np.linalg.norm(d, axis=1, keepdims=True)
        X = X_std + d * np.minimum(1.0, epsilon / (nrm + 1e-12))
    return X

def fit_noiseaug_logreg(Xtr, ytr, sigma=DEF_SIGMA, n_aug=3, C=1.0, seed=0):
    rng = np.random.default_rng(seed); Xs = [Xtr]; ys = [ytr]
    for _ in range(n_aug):
        Xs.append(Xtr + rng.normal(0, sigma, Xtr.shape)); ys.append(ytr)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(np.vstack(Xs), np.concatenate(ys))
    return clf.coef_[0], float(clf.intercept_[0]), clf

def smooth_proba(clf, X_std, sigma=DEF_SIGMA, n_draws=16, seed=0):
    rng = np.random.default_rng(seed); acc = np.zeros(len(X_std))
    for _ in range(n_draws):
        acc += clf.predict_proba(X_std + rng.normal(0, sigma, X_std.shape))[:, 1]
    return acc / n_draws

def oof_under(X, y, groups, eps, mode, seeds=range(N_SEEDS), n_splits=N_SPLITS):
    """Seed-averaged OOF prob per patient. mode in {clean, natural, adv, mitig}."""
    y = np.asarray(y).astype(int); groups = np.asarray(groups)
    acc = np.zeros(len(y)); cnt = np.zeros(len(y))
    for s in seeds:
        sgkf = StratifiedGroupKFold(n_splits, shuffle=True, random_state=s)
        oof = np.full(len(y), np.nan)
        for tr, te in sgkf.split(X, y, groups):
            sc = StandardScaler().fit(X[tr]); Xtr = sc.transform(X[tr]); Xte = sc.transform(X[te])
            if mode == "mitig":
                w, b, clf = fit_noiseaug_logreg(Xtr, y[tr], seed=s)
                Xadv = pgd_linear_l2(Xte, y[te], w, b, eps)
                oof[te] = smooth_proba(clf, Xadv, seed=s)
            else:
                clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(Xtr, y[tr])
                w, b = clf.coef_[0], float(clf.intercept_[0])
                if mode == "clean":
                    oof[te] = clf.predict_proba(Xte)[:, 1]
                elif mode == "natural":
                    oof[te] = clf.predict_proba(Xte + np.random.default_rng(s).normal(0, DEF_SIGMA, Xte.shape))[:, 1]
                elif mode == "adv":
                    oof[te] = clf.predict_proba(pgd_linear_l2(Xte, y[te], w, b, eps))[:, 1]
        acc += oof; cnt += 1
    return acc / cnt
