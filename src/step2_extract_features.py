"""Step 2: per-clip features -> patient matrix (cached)."""
import pandas as pd
from .config import MANIFEST_CSV, CLIP_FEATURES, PATIENT_MATRIX, PATIENT_ID_COL
from .features import extract_clip_features, aggregate_to_patient

if __name__ == "__main__":
    man = pd.read_csv(MANIFEST_CSV, dtype={PATIENT_ID_COL: str})
    recs = []
    for _, r in man.iterrows():
        f = extract_clip_features(r["audio_path"])
        if f is not None:
            f[PATIENT_ID_COL] = r[PATIENT_ID_COL]; recs.append(f)
    clip_df = pd.DataFrame(recs); clip_df.to_parquet(CLIP_FEATURES, index=False)
    pat = aggregate_to_patient(clip_df); pat.to_parquet(PATIENT_MATRIX, index=False)
    print(f"clips: {clip_df.shape} | patient matrix: {pat.shape}")
