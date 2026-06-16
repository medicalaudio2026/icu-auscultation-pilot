"""Step 5: robustness curve + paired tests (Table S2, Figure 3 data)."""
import os, numpy as np, pandas as pd
from .config import (PATIENT_MATRIX, LABELS_CSV, RESULTS_DIR, PATIENT_ID_COL, OUTCOME_COL)
from .robustness import oof_under, DEF_SIGMA
from .stats import delong_p, paired_boot_dauc, holm
from sklearn.metrics import roc_auc_score

EPS_FRACS = [0.0, 0.05, 0.10, 0.20, 0.40]; OP_FRAC = 0.10

if __name__ == "__main__":
    pat = pd.read_parquet(PATIENT_MATRIX); pat[PATIENT_ID_COL] = pat[PATIENT_ID_COL].astype(str)
    lab = pd.read_csv(LABELS_CSV, dtype={PATIENT_ID_COL: str})
    df = pat.merge(lab[[PATIENT_ID_COL, OUTCOME_COL]], on=PATIENT_ID_COL, how="inner")
    cols = [c for c in pat.columns if c != PATIENT_ID_COL]
    X = df[cols].to_numpy(float); y = df[OUTCOME_COL].to_numpy().astype(int); g = df[PATIENT_ID_COL].to_numpy()
    typ = np.sqrt(X.shape[1])

    clean = oof_under(X, y, g, 0.0, "clean")
    rows = []
    for fr in EPS_FRACS:
        eps = fr * typ
        und = clean if fr == 0 else oof_under(X, y, g, eps, "adv")
        dfd = oof_under(X, y, g, eps, "mitig")
        rows.append(dict(eps_frac=fr, AUROC_undef=roc_auc_score(y, und), AUROC_def=roc_auc_score(y, dfd)))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "robustness_curve.csv"), index=False)

    eps = OP_FRAC * typ
    conds = {"clean": clean, "natural": oof_under(X, y, g, DEF_SIGMA, "natural"),
             "adversarial": oof_under(X, y, g, eps, "adv"), "mitigated": oof_under(X, y, g, eps, "mitig")}
    comps = [("clean", "natural"), ("clean", "adversarial"), ("adversarial", "mitigated"), ("clean", "mitigated")]
    raw, recs = [], []
    for a, b in comps:
        _, _, p = delong_p(y, conds[a], conds[b]); pt, lo, hi = paired_boot_dauc(y, conds[a], conds[b])
        raw.append(p); recs.append(dict(comparison=f"{a} vs {b}", dAUROC=pt, CI_low=lo, CI_high=hi, p_raw=p))
    for r, pa in zip(recs, holm(raw)): r["p_holm"] = pa
    pd.DataFrame(recs).to_csv(os.path.join(RESULTS_DIR, "robustness_paired_tests.csv"), index=False)
    print("robustness written")
