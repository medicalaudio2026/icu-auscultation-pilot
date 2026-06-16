# Digital auscultation for ICU outcome prediction: signal localization and adversarial vulnerability

Reproducible analysis code for the Rapid Communication of the same title.

This repository builds **lung-sound-only** machine-learning models to discriminate
**intensive care unit (ICU) outcome** (death vs survival) from digital lung-sound
recordings, using the public **CoCross COVID-19 ICU Lung Sounds** database
(173 mechanically ventilated adults with COVID-19 acute respiratory distress
syndrome). It reproduces the three analyses reported in the paper:

1. Discrimination of lung-sound, clinical, and fusion models (patient-grouped,
   five-seed cross-validation with bootstrap confidence intervals).
2. Localization of the prognostic signal to whole-clip spectral versus
   adventitious-event features, on an independent precomputed feature set.
3. Robustness of the sound-only model to additive noise and to white-box,
   feature-space adversarial perturbation, with an adaptively evaluated mitigation.

> **Author-blinded.** This copy is anonymized for peer review. It contains no
> author, institution, or identifying information.

## Data

The CoCross COVID-19 ICU Lung Sounds database is publicly available on Figshare
and is **not redistributed here**. Download it from the source listed in the
manuscript, and set `DATA_DIR` in `src/config.py` to the unzipped location.
The database includes the per-clip audio, a clinical/outcome spreadsheet, and a
sheet of precomputed acoustic features used for the independent replication.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.11. Core dependencies: numpy, pandas, scipy, scikit-learn,
librosa, soundfile, matplotlib, pyarrow.

## Reproduce

Run in order from the repository root:

```bash
python -m src.step1_build_manifest    # index lung-only clips + read labels from the database file
python -m src.step2_extract_features  # per-clip features -> patient matrix (cached)
python -m src.step3_evaluate          # Table 1, Table S1, Figure 1, OOF predictions
python -m src.step4_replication       # authors' features, PCA, mechanism (Table S3, Figure 2)
python -m src.step5_robustness        # noise/adversarial/mitigation (Table S2, Figure 3)
python -m figures.make_figures        # regenerate publication figures from cached results
```

All outputs are written to `results/`. Each step caches intermediate artefacts,
so reruns are fast.

## Methods summary (matches the manuscript)

- **Unit of analysis:** patient. Clips are aggregated to one feature vector per
  patient (mean, standard deviation, 10th and 90th percentiles) before modelling.
- **Audio features:** loaded with librosa at 4 kHz, mono; whole clip, no
  segmentation; recordings < 0.5 s discarded; spectral features from a 512-point
  short-time Fourier transform; 13 mel-frequency cepstral coefficients.
- **Cross-validation:** StratifiedGroupKFold (stratify on outcome, group on
  patient), 5 folds x 5 seeds; all preprocessing fit inside training folds.
- **Models:** regularized logistic regression, random forest, histogram gradient
  boosting.
- **Discrimination:** AUROC with 2,000-sample patient-level bootstrap 95% CIs.
- **Localization:** independent precomputed feature set, in-fold PCA(30);
  whole-clip spectral vs adventitious-event subsets compared with the DeLong test.
- **Robustness:** white-box L2 projected-gradient-descent in feature space,
  budget as a fraction of the feature-vector norm; mitigation = noise-augmented
  training + randomized smoothing, evaluated adaptively; paired DeLong with Holm
  correction.

## License

Released for peer review. A permissive open-source license will be attached on
publication.
