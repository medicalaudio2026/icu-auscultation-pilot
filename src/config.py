"""Central configuration: paths and parameters. Edit DATA_DIR to your unzipped CoCross folder."""
import os

# ---- paths ----
DATA_DIR    = os.environ.get("COCROSS_DATA_DIR", "./data/CoCross")  # unzipped database root
RESULTS_DIR = "./results"
CACHE_DIR   = os.path.join(RESULTS_DIR, "cache")
for d in (RESULTS_DIR, CACHE_DIR):
    os.makedirs(d, exist_ok=True)

MANIFEST_CSV        = os.path.join(CACHE_DIR, "manifest.csv")
LABELS_CSV          = os.path.join(CACHE_DIR, "labels.csv")
CLIP_FEATURES       = os.path.join(CACHE_DIR, "clip_features.parquet")
PATIENT_MATRIX      = os.path.join(CACHE_DIR, "patient_matrix.parquet")
AUTHORS_MATRIX      = os.path.join(CACHE_DIR, "authors_patient_matrix.parquet")

# ---- columns ----
OUTCOME_COL     = "died_icu"        # 1 = died in ICU, 0 = survived
PATIENT_ID_COL  = "patient_id"
CLINICAL_COLS   = ["age", "sofa", "apache_ii", "charlson"]

# ---- audio ----
TARGET_SR = 4000          # Hz; lung-sound energy concentrated below ~2 kHz
N_MFCC    = 13
N_FFT     = 512
MIN_SECONDS = 0.5         # discard clips shorter than this

# lung auscultation sites kept (heart sounds excluded)
LUNG_KEYS = ("LeftLungApexFront", "LeftLungBaseFront", "LeftLungBaseBack",
             "RightLungApexFront", "RightLungBaseFront", "RightLungBaseBack")

# ---- modelling ----
N_SPLITS, N_SEEDS, BASE_SEED, N_BOOT = 5, 5, 0, 2000
