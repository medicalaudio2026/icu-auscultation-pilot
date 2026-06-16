"""Per-clip lung-sound feature extraction and patient-level aggregation.

Audio is loaded with librosa at TARGET_SR (4 kHz), mono, whole clip (no
segmentation). Recordings shorter than MIN_SECONDS are discarded. Spectral
features use an N_FFT-point STFT; N_MFCC mel-frequency cepstral coefficients
are extracted. Per-patient aggregation = mean, std, 10th and 90th percentiles.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from .config import TARGET_SR, N_MFCC, N_FFT, MIN_SECONDS, PATIENT_ID_COL

def extract_clip_features(path: str, sr: int = TARGET_SR) -> dict | None:
    import librosa
    try:
        y, sr = librosa.load(path, sr=sr, mono=True)
    except Exception:
        return None
    if y is None or len(y) < int(sr * MIN_SECONDS):
        return None
    eps = 1e-10
    f: dict[str, float] = {}
    # time-domain
    f["zcr_mean"] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    f["zcr_std"]  = float(np.std(librosa.feature.zero_crossing_rate(y)))
    rms = librosa.feature.rms(y=y)[0]
    f["rms_mean"], f["rms_std"] = float(rms.mean()), float(rms.std())
    # spectral shape
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bw   = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    flat = librosa.feature.spectral_flatness(y=y)[0]
    f["centroid_mean"], f["centroid_std"] = float(cent.mean()), float(cent.std())
    f["bandwidth_mean"], f["bandwidth_std"] = float(bw.mean()), float(bw.std())
    f["rolloff_mean"], f["rolloff_std"] = float(roll.mean()), float(roll.std())
    f["flatness_mean"], f["flatness_std"] = float(flat.mean()), float(flat.std())
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, fmin=100, n_bands=4)
    for i in range(contrast.shape[0]):
        f[f"contrast{i}_mean"] = float(contrast[i].mean())
        f[f"contrast{i}_std"]  = float(contrast[i].std())
    # band-energy ratios (100-2000 Hz)
    S = np.abs(librosa.stft(y, n_fft=N_FFT))**2 + eps
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    total = S.sum() + eps
    band = lambda lo, hi: float(S[(freqs >= lo) & (freqs < hi)].sum())
    f["energy_100_500"]   = band(100, 500) / total
    f["energy_500_1000"]  = band(500, 1000) / total
    f["energy_1000_2000"] = band(1000, 2000) / total
    # spectral entropy / irregularity
    ps = S.sum(axis=1); ps = ps / (ps.sum() + eps)
    f["spec_entropy"] = float(-(ps * np.log(ps + eps)).sum())
    f["spec_irregularity_mean"] = float(np.mean(np.abs(np.diff(S, axis=0)).sum(axis=0) / total))
    f["spec_irregularity_p10"]  = float(np.percentile(np.abs(np.diff(S, axis=0)).sum(axis=0) / total, 10))
    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(mfcc.shape[0]):
        f[f"mfcc{i}_mean"] = float(mfcc[i].mean())
        f[f"mfcc{i}_std"]  = float(mfcc[i].std())
    return f

def aggregate_to_patient(clip_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-clip features to one row per patient: mean, std, p10, p90."""
    feat_cols = [c for c in clip_df.columns if c != PATIENT_ID_COL]
    g = clip_df.groupby(PATIENT_ID_COL)[feat_cols]
    out = g.agg(["mean", "std",
                 lambda s: s.quantile(0.10),
                 lambda s: s.quantile(0.90)])
    out.columns = [f"{c}_{stat}" for c, stat in
                   zip([c for c in feat_cols for _ in range(4)],
                       ["mean", "std", "p10", "p90"] * len(feat_cols))]
    out["n_clips"] = g.size()
    return out.reset_index()
