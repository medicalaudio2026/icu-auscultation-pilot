"""Step 3: main discrimination (Table 1, Table S1, OOF predictions)."""
import os, pandas as pd, numpy as np
from .config import (PATIENT_MATRIX, LABELS_CSV, RESULTS_DIR, PATIENT_ID_COL,
                     OUTCOME_COL, CLINICAL_COLS)
from .modeling import evaluate, summarize

if __name__ == "__main__":
    pat = pd.read_parquet(PATIENT_MATRIX); pat[PATIENT_ID_COL] = pat[PATIENT_ID_COL].astype(str)
    lab = pd.read_csv(LABELS_CSV, dtype={PATIENT_ID_COL: str})
    df = pat.merge(lab, on=PATIENT_ID_COL, how="inner")
    audio_cols = [c for c in pat.columns if c != PATIENT_ID_COL]
    sets = {"audio": audio_cols, "clinical": CLINICAL_COLS, "fusion": audio_cols + CLINICAL_COLS}
    all_sum, oof_rows = [], []
    for name, cols in sets.items():
        ev = evaluate(df, df[OUTCOME_COL], df[PATIENT_ID_COL], cols, name)
        all_sum.append(summarize(ev))
        for m, r in ev.items():
            oof_rows.append(pd.DataFrame({"feature_set": name, "model": m,
                PATIENT_ID_COL: df[PATIENT_ID_COL].values, "y_true": r["y"], "oof_prob": r["oof"]}))
    summ = pd.concat(all_sum, ignore_index=True)
    summ.to_csv(os.path.join(RESULTS_DIR, "metrics_summary.csv"), index=False)
    pd.concat(oof_rows, ignore_index=True).to_csv(os.path.join(RESULTS_DIR, "oof_predictions.csv"), index=False)
    print(summ[["feature_set","model","AUROC_mean","AUROC_ci_low","AUROC_ci_high"]].to_string(index=False))
