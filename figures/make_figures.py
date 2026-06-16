"""
Publication figure generator for the Rapid Communication.
ICU lung-sound mortality pilot.

USAGE (Colab):
  1. Set RESULTS below to your results directory, e.g.
     RESULTS = "./results"
  2. Ensure these CSVs exist there:
     oof_predictions.csv, metrics_paper.csv,
     metrics_authors_mechanism.csv, robustness_curve.csv
  3. Run. Produces, in RESULTS:
     Figure1_roc_calibration.{png,pdf}
     Figure2_mechanism.{png,pdf}
     Figure3_robustness.{png,pdf}

Style: Times New Roman (serif fallback DejaVu Serif), Okabe-Ito colorblind-safe
palette, 300 dpi, plain-language axis labels. AUROC values in the Figure 1 legend
are read from metrics_paper.csv so they match Table 1 exactly (the curve itself is
pooled out-of-fold; the reported AUROC is the seed-averaged value with CI).

NOTE on Times New Roman in Colab: the metric font may not be installed. To get true
Times New Roman, run once:
  !apt-get -qq install -y ttf-mscorefonts-installer fonts-liberation > /dev/null
  import matplotlib.font_manager as fm; fm._load_fontmanager(try_read_cache=False)
Otherwise matplotlib falls back to a near-identical serif face, which is fine for review.
"""
"""Publication figures for the Rapid Communication. Reads the result CSVs and renders
Figure 1 (ROC + calibration), Figure 2 (mechanism), Figure 3 (robustness).
Times New Roman, colorblind-safe palette, clear axis labels, 300 dpi."""
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve

# ---- global style: serif (Times), readable sizes, colorblind-safe ----
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
    "axes.linewidth": 0.8, "savefig.dpi": 300, "figure.dpi": 120,
})
# Okabe-Ito colorblind-safe palette
C = {"audio":"#0072B2", "clinical":"#D55E00", "fusion":"#009E73",
     "undef":"#D55E00", "def":"#0072B2", "whole":"#0072B2", "event":"#999999"}

RESULTS = "./results"

# ============ FIGURE 1: ROC + calibration ============
oof = pd.read_csv(f"{RESULTS}/oof_predictions.csv")
paper = pd.read_csv(f"{RESULTS}/metrics_paper.csv")
# models reported in Table 1 (best per set)
pick = {"audio":"hgb", "clinical":"logreg", "fusion":"hgb"}
label = {"audio":"Lung sound", "clinical":"Clinical", "fusion":"Fusion"}
def table1_auroc(fs, m):
    s = paper[(paper.feature_set==fs)&(paper.model==m)]["AUROC"].iloc[0]  # e.g. "0.697 (0.651-0.808)"
    return s

fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.9))
for fs, m in pick.items():
    d = oof[(oof.feature_set==fs) & (oof.model==m)]
    fpr, tpr, _ = roc_curve(d.y_true, d.oof_prob)
    ax[0].plot(fpr, tpr, color=C[fs], lw=1.8,
               label=f"{label[fs]} (AUROC {table1_auroc(fs,m).split(' ')[0]})")
ax[0].plot([0,1],[0,1], color="0.6", lw=0.9, ls=(0,(4,3)))
ax[0].set_xlabel("False positive rate (1 − specificity)")
ax[0].set_ylabel("True positive rate (sensitivity)")
ax[0].set_title("(a) Discrimination")
ax[0].set_xlim(-0.02,1.02); ax[0].set_ylim(-0.02,1.02)
ax[0].legend(loc="lower right", frameon=False)
ax[0].set_aspect("equal")
for fs, m in pick.items():
    d = oof[(oof.feature_set==fs) & (oof.model==m)]
    frac_pos, mean_pred = calibration_curve(d.y_true, d.oof_prob, n_bins=5, strategy="quantile")
    ax[1].plot(mean_pred, frac_pos, marker="o", ms=4, color=C[fs], lw=1.5, label=label[fs])
ax[1].plot([0,1],[0,1], color="0.6", lw=0.9, ls=(0,(4,3)))
ax[1].set_xlabel("Mean predicted probability")
ax[1].set_ylabel("Observed mortality fraction")
ax[1].set_title("(b) Calibration")
ax[1].set_xlim(-0.02,1.02); ax[1].set_ylim(-0.02,1.02)
ax[1].legend(loc="upper left", frameon=False)
ax[1].set_aspect("equal")
fig.tight_layout()
fig.savefig("Figure1_roc_calibration.png", bbox_inches="tight")
fig.savefig("Figure1_roc_calibration.pdf", bbox_inches="tight")
print("Figure 1 legend AUROCs (from Table 1):",
      {fs: table1_auroc(fs,m).split(' ')[0] for fs,m in pick.items()})

# ============ FIGURE 2: mechanism (whole-clip vs event) ============
mech = pd.read_csv(f"{RESULTS}/metrics_authors_mechanism.csv")
order = ["whole_clip_spectral","event_crackle_wheeze_squawk"]
names = {"whole_clip_spectral":"Whole-clip\nspectral",
         "event_crackle_wheeze_squawk":"Adventitious-event\n(crackle/wheeze/squawk)"}
mech = mech.set_index("feature_group").loc[order].reset_index()
auc = mech.AUROC_mean.values
lo  = auc - mech.AUROC_ci_low.values
hi  = mech.AUROC_ci_high.values - auc
fig, ax = plt.subplots(figsize=(5.0, 3.8))
ypos = np.arange(len(order))[::-1]
bars = ax.barh(ypos, auc, xerr=[lo,hi], height=0.5,
               color=[C["whole"], C["event"]], capsize=4,
               error_kw=dict(lw=1, capthick=1))
ax.axvline(0.5, color="0.5", lw=0.9, ls=(0,(4,3)))
ax.text(0.5, -0.62, "chance", color="0.4", ha="center", va="top", fontsize=8.5)
ax.set_yticks(ypos); ax.set_yticklabels([names[o] for o in order])
ax.set_xlabel("AUROC (95% CI)")
ax.set_xlim(0.40, 0.82)
for y,a in zip(ypos,auc):
    ax.text(a+0.005, y+0.28, f"{a:.3f}", va="center", fontsize=9)
ax.set_title("ICU-outcome discrimination by acoustic feature group")
fig.tight_layout()
fig.savefig("Figure2_mechanism.png", bbox_inches="tight")
fig.savefig("Figure2_mechanism.pdf", bbox_inches="tight")
print("Figure 2 written. whole=%.3f event=%.3f" % (auc[0], auc[1]))

# ============ FIGURE 3: robustness curve ============
rc = pd.read_csv(f"{RESULTS}/robustness_curve.csv").sort_values("eps_frac")
fig, ax = plt.subplots(figsize=(5.4, 4.0))
ax.errorbar(rc.eps_frac*100, rc.AUROC_undef_mean, yerr=rc.AUROC_undef_sd,
            marker="o", ms=5, lw=1.6, color=C["undef"], capsize=3, label="Undefended")
ax.errorbar(rc.eps_frac*100, rc.AUROC_def_mean, yerr=rc.AUROC_def_sd,
            marker="s", ms=5, lw=1.6, color=C["def"], ls=(0,(5,2)), capsize=3,
            label="Defended (noise-aug + smoothing)")
ax.axhline(0.5, color="0.5", lw=0.9, ls=(0,(4,3)))
ax.text(rc.eps_frac.max()*100, 0.5, " chance", color="0.4", va="bottom", ha="right", fontsize=8.5)
ax.set_xlabel("Adversarial perturbation budget (% of feature-vector L2 norm)")
ax.set_ylabel("AUROC (mean ± SD over 5 seeds)")
ax.set_ylim(-0.03, 0.78)
ax.set_title("Adversarial robustness of the lung-sound model")
ax.legend(loc="upper right", frameon=False)
fig.tight_layout()
fig.savefig("Figure3_robustness.png", bbox_inches="tight")
fig.savefig("Figure3_robustness.pdf", bbox_inches="tight")
print("Figure 3 written. clean=%.3f adv@10%%=%.3f def@10%%=%.3f" %
      (rc[rc.eps_frac==0.0].AUROC_undef_mean.iloc[0],
       rc[rc.eps_frac==0.1].AUROC_undef_mean.iloc[0],
       rc[rc.eps_frac==0.1].AUROC_def_mean.iloc[0]))
