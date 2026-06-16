"""Correlated-ROC statistics: DeLong test and Holm correction."""
import numpy as np
from scipy import stats as _st

def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    T2 = np.empty(N); T2[J] = T; return T2

def _fast_delong(ps, m):
    k = ps.shape[0]; n = ps.shape[1] - m
    pos, neg = ps[:, :m], ps[:, m:]
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(pos[r]); ty[r] = _midrank(neg[r]); tz[r] = _midrank(ps[r])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1) / 2 / n
    v01 = (tz[:, :m] - tx) / n; v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov

def delong_p(y, s1, s2):
    """Two-sided DeLong p-value for AUROC(s1) vs AUROC(s2) on the same labels y."""
    y = np.asarray(y).astype(int); o = (-y).argsort(kind="mergesort"); m = int(y.sum())
    ps = np.vstack((np.asarray(s1, float)[o], np.asarray(s2, float)[o]))
    a, c = _fast_delong(ps, m); vd = c[0, 0] + c[1, 1] - 2 * c[0, 1]
    if vd <= 0:
        return float(a[0]), float(a[1]), (1.0 if abs(a[0] - a[1]) < 1e-12 else 0.0)
    z = (a[0] - a[1]) / np.sqrt(vd)
    return float(a[0]), float(a[1]), float(2 * _st.norm.sf(abs(z)))

def holm(pvals):
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * p[idx]); adj[idx] = min(1.0, run)
    return adj

def paired_boot_dauc(y, s1, s2, n_boot=3000, seed=1):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed); y = np.asarray(y); s1 = np.asarray(s1); s2 = np.asarray(s2)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2: continue
        diffs.append(roc_auc_score(y[idx], s1[idx]) - roc_auc_score(y[idx], s2[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), float(lo), float(hi)
