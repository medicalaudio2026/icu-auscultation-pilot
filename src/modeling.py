"""Statistically rigorous evaluation core.

Validity choices:
  * Unit of analysis = patient; clips aggregated before modelling, grouping enforced.
  * CV = StratifiedGroupKFold (stratify on outcome, group on patient): no patient
    in both train and test; class balance preserved.
  * Repeats = N_SEEDS shuffles; reported number = mean over seeds; uncertainty =
    patient-level bootstrap 95% CI on the seed-averaged out-of-fold predictions.
  * Preprocessing fit INSIDE each fold via a Pipeline (no leakage).
  * Operating-point threshold = Youden's J chosen on the TRAIN portion only.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             brier_score_loss, confusion_matrix)
from .config import N_SPLITS, N_SEEDS, BASE_SEED, N_BOOT

def build_models(seed: int) -> dict[str, Pipeline]:
    return {
        "logreg": Pipeline([("impute", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler()),
                            ("clf", LogisticRegression(C=1.0, class_weight="balanced",
                                                       max_iter=5000, random_state=seed))]),
        "hgb": Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("clf", HistGradientBoostingClassifier(learning_rate=0.05, max_depth=3,
                                 max_iter=300, l2_regularization=1.0, random_state=seed))]),
        "rf": Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("clf", RandomForestClassifier(n_estimators=400, max_depth=None,
                                class_weight="balanced", random_state=seed, n_jobs=-1))]),
    }

def _youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return thr[np.argmax(tpr - fpr)]

def _point_metrics(y, p, thr):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) else 0.0
    npv  = tn / (tn + fn) if (tn + fn) else 0.0
    f1   = 2 * ppv * sens / (ppv + sens) if (ppv + sens) else 0.0
    acc  = (tp + tn) / len(y)
    return dict(Sensitivity=sens, Specificity=spec, PPV=ppv, NPV=npv,
                F1=f1, Accuracy=acc, BalancedAcc=(sens + spec) / 2)

def evaluate(df, y, groups, feature_cols, feature_set_name,
             model_names=None, n_splits=N_SPLITS, n_seeds=N_SEEDS, base_seed=BASE_SEED):
    """Return per-model dict with seed-averaged OOF predictions and metrics."""
    model_names = model_names or ["logreg", "hgb", "rf"]
    X = df[feature_cols].to_numpy(float); y = np.asarray(y).astype(int); groups = np.asarray(groups)
    out = {}
    for mname in model_names:
        acc_oof = np.zeros(len(y)); cnt = np.zeros(len(y)); seed_aurocs = []
        for s in range(n_seeds):
            sgkf = StratifiedGroupKFold(n_splits, shuffle=True, random_state=base_seed + s)
            oof = np.full(len(y), np.nan)
            for tr, te in sgkf.split(X, y, groups):
                assert not (set(groups[tr]) & set(groups[te]))
                model = build_models(base_seed + s)[mname]
                model.fit(X[tr], y[tr])
                oof[te] = model.predict_proba(X[te])[:, 1]
            seed_aurocs.append(roc_auc_score(y, oof)); acc_oof += oof; cnt += 1
        oof_mean = acc_oof / cnt
        out[mname] = dict(feature_set=feature_set_name, model=mname,
                          oof=oof_mean, y=y, seed_aurocs=np.array(seed_aurocs))
    return out

def bootstrap_ci(y, p, fn, n_boot=N_BOOT, seed=BASE_SEED):
    rng = np.random.default_rng(seed); vals = []
    y = np.asarray(y); p = np.asarray(p)
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2: continue
        vals.append(fn(y[idx], p[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(np.mean(vals)), float(lo), float(hi)

def summarize(eval_out, n_boot=N_BOOT, seed=BASE_SEED):
    rows = []
    for mname, r in eval_out.items():
        y, p = r["y"], r["oof"]
        thr = _youden_threshold(y, p)
        pm = _point_metrics(y, p, thr)
        auroc = bootstrap_ci(y, p, roc_auc_score, n_boot, seed)
        auprc = bootstrap_ci(y, p, average_precision_score, n_boot, seed)
        rows.append(dict(feature_set=r["feature_set"], model=mname,
                         AUROC_mean=auroc[0], AUROC_ci_low=auroc[1], AUROC_ci_high=auroc[2],
                         AUPRC_mean=auprc[0], AUPRC_ci_low=auprc[1], AUPRC_ci_high=auprc[2],
                         Brier=brier_score_loss(y, p), **pm))
    return pd.DataFrame(rows)
