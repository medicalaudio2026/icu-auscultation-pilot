"""Step 1: index lung-only clips and build labels from the CoCross database file."""
import os, glob, re
import numpy as np, pandas as pd
from .config import DATA_DIR, MANIFEST_CSV, LABELS_CSV, LUNG_KEYS, PATIENT_ID_COL, OUTCOME_COL

def build_manifest():
    wavs = sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.wav"), recursive=True))
    rows = []
    for p in wavs:
        fn = os.path.basename(p)
        if not any(k in fn for k in LUNG_KEYS):      # keep lung sites; drop heart sounds
            continue
        m = re.match(r"(\d+)_", fn)
        if m:
            rows.append({PATIENT_ID_COL: m.group(1), "audio_path": p})
    man = pd.DataFrame(rows); man.to_csv(MANIFEST_CSV, index=False)
    print(f"Lung clips: {len(man)} | patients: {man[PATIENT_ID_COL].nunique()}")
    return man

def build_labels():
    xlsx = glob.glob(os.path.join(DATA_DIR, "**", "*.xlsx"), recursive=True)[0]
    data = pd.ExcelFile(xlsx).parse("Data"); data["ID"] = data["ID"].astype(str)
    first = lambda s: (s.dropna().iloc[0] if s.dropna().size else np.nan)
    agg = data.groupby("ID").agg(
        icu_outcome=("ICU Outcome", first), age=("Age", first),
        sofa=("SOFA", first), apache_ii=("APACHE II", first),
        charlson=("Charlson Comorbidity index", first)).reset_index().rename(columns={"ID": PATIENT_ID_COL})
    agg[OUTCOME_COL] = (agg["icu_outcome"].astype(str).str.strip().str.lower() == "death").astype(int)
    labels = agg[[PATIENT_ID_COL, OUTCOME_COL, "age", "sofa", "apache_ii", "charlson"]]
    labels.to_csv(LABELS_CSV, index=False)
    print(f"labels: {labels.shape} | prevalence {labels[OUTCOME_COL].mean():.3f}")
    return labels

if __name__ == "__main__":
    build_manifest(); build_labels()
