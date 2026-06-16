"""Step 4: independent precomputed-feature replication, PCA, and mechanism test
(whole-clip spectral vs adventitious-event), with DeLong significance."""
import os, glob, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from .config import (DATA_DIR, LABELS_CSV, PATIENT_MATRIX, RESULTS_DIR,
                     PATIENT_ID_COL, OUTCOME_COL, N_SPLITS, N_SEEDS, BASE_SEED, N_BOOT)
from .stats import delong_p

EVENT_KEYS = ("crackle", "wheeze", "squawk")
def is_event(c): return any(k in c.lower() for k in EVENT_KEYS)
def is_whole(c):
    cl = c.lower()
    return ("win64" in cl or "_spect" in cl or "mel" in cl) and not is_event(c)

def build_authors_matrix():
    xlsx = glob.glob(os.path.join(DATA_DIR, "**", "*.xlsx"), recursive=True)[0]
    auth = pd.ExcelFile(xlsx).parse("Extracted Sound Features"); auth["ID"] = auth["ID"].astype(str)
    feat = [c for c in auth.columns if c not in ("ID", "Date")]
    for c in feat: auth[c] = pd.to_numeric(auth[c], errors="coerce")
    g = auth.groupby("ID")[feat]; A = g.agg(["mean", "std"])
    A.columns = [f"{a}__{b}" for a, b in A.columns]; A = A.reset_index().rename(columns={"ID": PATIENT_ID_COL})
    std_cols = [c for c in A.columns if c.endswith("__std")]; A[std_cols] = A[std_cols].fillna(0.0)
    mean_cols = [c for c in A.columns if c.endswith("__mean")]; A[mean_cols] = A[mean_cols].fillna(A[mean_cols].median())
    val = [c for c in A.columns if c != PATIENT_ID_COL]
    A = A.drop(columns=[c for c in val if A[c].nunique() <= 1])
    return A

def eval_logreg(df, cols, pca=None):
    X = df[cols].to_numpy(float); y = df[OUTCOME_COL].to_numpy().astype(int); g = df[PATIENT_ID_COL].to_numpy()
    acc = np.zeros(len(y)); cnt = np.zeros(len(y))
    for s in range(N_SEEDS):
        sgkf = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=BASE_SEED + s)
        oof = np.full(len(y), np.nan)
        for tr, te in sgkf.split(X, y, g):
            steps = [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
            if pca: steps.append(("pca", PCA(n_components=min(pca, len(cols)), random_state=BASE_SEED + s)))
            steps.append(("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)))
            pipe = Pipeline(steps); pipe.fit(X[tr], y[tr]); oof[te] = pipe.predict_proba(X[te])[:, 1]
        acc += oof; cnt += 1
    oof_mean = acc / cnt
    rng = np.random.default_rng(12345); bs = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2: continue
        bs.append(roc_auc_score(y[idx], oof_mean[idx]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return roc_auc_score(y, oof_mean), lo, hi, oof_mean

if __name__ == "__main__":
    A = build_authors_matrix()
    lab = pd.read_csv(LABELS_CSV, dtype={PATIENT_ID_COL: str})
    pat = pd.read_parquet(PATIENT_MATRIX); keep = set(pat[PATIENT_ID_COL].astype(str))
    df = A.merge(lab[[PATIENT_ID_COL, OUTCOME_COL]], on=PATIENT_ID_COL, how="inner")
    df[PATIENT_ID_COL] = df[PATIENT_ID_COL].astype(str); df = df[df[PATIENT_ID_COL].isin(keep)].copy()
    df[OUTCOME_COL] = df[OUTCOME_COL].astype(int)
    feat = [c for c in df.columns if c not in (PATIENT_ID_COL, OUTCOME_COL)]
    whole = [c for c in feat if is_whole(c)]; event = [c for c in feat if is_event(c)]

    raw = eval_logreg(df, feat); pca = eval_logreg(df, feat, pca=30)
    w = eval_logreg(df, whole, pca=30); e = eval_logreg(df, event, pca=30)
    _, _, p = delong_p(df[OUTCOME_COL].to_numpy().astype(int), w[3], e[3])

    pd.DataFrame([
        {"analysis": "raw", "n_features": len(feat), "AUROC": raw[0], "ci_low": raw[1], "ci_high": raw[2]},
        {"analysis": "pca30", "n_features": len(feat), "AUROC": pca[0], "ci_low": pca[1], "ci_high": pca[2]},
        {"analysis": "whole_clip_spectral", "n_features": len(whole), "AUROC": w[0], "ci_low": w[1], "ci_high": w[2]},
        {"analysis": "event", "n_features": len(event), "AUROC": e[0], "ci_low": e[1], "ci_high": e[2]},
    ]).to_csv(os.path.join(RESULTS_DIR, "metrics_replication.csv"), index=False)
    print(f"whole {w[0]:.3f} vs event {e[0]:.3f}  DeLong P={p:.4g}")
