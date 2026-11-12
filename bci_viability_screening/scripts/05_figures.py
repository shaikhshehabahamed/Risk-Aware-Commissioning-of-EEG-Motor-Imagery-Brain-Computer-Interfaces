"""
Script 05: Generate all paper figures.

Figures:
  Fig 1: Label distribution + label noise ceiling analysis
  Fig 2: Top spectral feature Spearman scatter plots (spectral entropy, parietal alpha)
  Fig 3: AUC comparison bar chart (all feature sets × classifiers + nested)
  Fig 4: ROC curve for best model (SP-aug-68/LR-nested-top28, AUC=0.682)
         + clear subgroup ROC (SP-47/LR-nested-top10, AUC=0.685)
  Fig 5: Microstate templates (topographies) with temporal statistics
  Fig 6: Feature importance heatmap (Spearman |rho| for top features)

Reads from: results/ directory
"""

import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

try:
    _HERE = Path(__file__).resolve().parent
except NameError:
    _HERE = Path(r"C:\Users\sheha\Downloads\resting_bci_screening\scripts")

PROJ    = _HERE.parent
RESULTS = PROJ / "results"
FIGURES = PROJ / "figures"
FIGURES.mkdir(exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size":   11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

BLUE   = "#2563EB"
ORANGE = "#F59E0B"
GREEN  = "#10B981"
RED    = "#EF4444"
PURPLE = "#8B5CF6"
GRAY   = "#6B7280"

# ── Load data ─────────────────────────────────────────────────────────────────
data     = np.load(RESULTS / "features.npz", allow_pickle=True)
subjects = list(data["subjects"])
X_sp     = data["X_sp"].astype(float)
X_ms     = data["X_ms"].astype(float)
X_eo     = data["X_eo"].astype(float)
sp_names = list(data["sp_names"]) if "sp_names" in data else [f"SP_{i}" for i in range(47)]

extra    = np.load(RESULTS / "features_extra.npz", allow_pickle=True)
X_extra  = extra["X_extra"].astype(float)
extra_names = list(extra["feature_names"]) if "feature_names" in extra else []
X_aug    = np.hstack([X_sp, X_extra])   # 68 features

with open(RESULTS / "viability_labels.json") as f:
    ld = json.load(f)
y         = np.array([int(r["viable"])    for r in ld["per_subject"]])
fold3_acc = np.array([r["fold3_accuracy"] for r in ld["per_subject"]])
n         = len(y)

# ── Helper: nested Spearman LOSO ──────────────────────────────────────────────
def loso_nested(X, y_bin, k, C=2.0):
    yt, yp = [], []
    for i in range(len(y_bin)):
        tr = [j for j in range(len(y_bin)) if j != i]
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[[i]])
        rhos = np.array([abs(spearmanr(Xtr[:, j], y_bin[tr])[0])
                         for j in range(Xtr.shape[1])])
        top  = np.argsort(rhos)[::-1][:k]
        lr   = LogisticRegression(solver="lbfgs", C=C, max_iter=2000)
        lr.fit(Xtr[:, top], y_bin[tr])
        yt.append(int(y_bin[i]))
        yp.append(float(lr.predict_proba(Xte[:, top])[0, 1]))
    return np.array(yt), np.array(yp)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Label distribution and noise ceiling
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 1: Label distribution...", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Panel A: histogram of fold-3 accuracy
ax = axes[0]
ax.hist(fold3_acc, bins=16, range=(0.2, 1.0), color=BLUE, alpha=0.8, edgecolor="white")
ax.axvline(0.60, color=RED, lw=2, ls="--", label="Viability threshold (60%)")
ax.axvspan(0.50, 0.70, alpha=0.10, color=ORANGE, label="Borderline region (±1 SE)")
ax.set_xlabel("Fold-3 FBCSP-LDA accuracy")
ax.set_ylabel("Number of subjects")
ax.set_title("A: Prospective viability label distribution (n=92)")
ax.legend(fontsize=9)
n_viable    = int(y.sum())
n_nonviable = n - n_viable
n_border    = int(((fold3_acc >= 0.50) & (fold3_acc < 0.70)).sum())
ax.text(0.97, 0.97, f"Viable: {n_viable}  Non-viable: {n_nonviable}\n"
                     f"Borderline (SE range): {n_border} ({100*n_border//n}%)",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Panel B: theoretical AUC ceiling vs SE
ax2 = axes[1]
# Simulate label noise effect on AUC ceiling
n_trials_range = np.arange(5, 100, 5)
ceilings = []
for n_tr in n_trials_range:
    se = np.sqrt(0.5 * 0.5 / n_tr)
    # Fraction borderline at SE width around 0.60
    from scipy.stats import norm
    border_frac = norm.cdf(0.60 + se) - norm.cdf(0.60 - se)
    # AUC ceiling approximation
    auc_ceil = 1.0 - 0.5 * border_frac
    ceilings.append(auc_ceil)
ax2.plot(n_trials_range, ceilings, color=BLUE, lw=2)
ax2.axvline(15, color=RED, ls="--", lw=1.5, label="~15 trials/class (this study)")
ax2.axhline(0.74, color=ORANGE, ls=":", lw=1.5, label="Estimated ceiling ≈ 0.74")
ax2.set_xlabel("Test trials per class (Fold-3)")
ax2.set_ylabel("Estimated AUC ceiling")
ax2.set_title("B: Label noise ceiling vs trial count")
ax2.set_ylim(0.5, 1.0)
ax2.legend(fontsize=9)
ax2.set_xlim(5, 100)

plt.tight_layout()
fig.savefig(FIGURES / "fig1_label_distribution.pdf", bbox_inches="tight")
fig.savefig(FIGURES / "fig1_label_distribution.png", bbox_inches="tight")
plt.close()
print("  Saved fig1_label_distribution", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Key feature Spearman scatter plots
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 2: Spearman scatter plots...", flush=True)

# Compute Spearman for all SP features
all_feat_names = list(sp_names) + list(extra_names)
X_all_aug = X_aug

rho_vals, p_vals = [], []
for j in range(X_all_aug.shape[1]):
    r, p = spearmanr(X_all_aug[:, j], fold3_acc)
    rho_vals.append(r)
    p_vals.append(p)

# Find spectral entropy index
spec_ent_idx = next((i for i, nm in enumerate(all_feat_names)
                     if "entropy" in nm.lower()), 0)

# Find parietal alpha indices
pz_idx  = next((i for i, nm in enumerate(all_feat_names) if "Pz" in nm and "alpha" in nm), None)
p4_idx  = next((i for i, nm in enumerate(all_feat_names) if "P4"  in nm and "alpha" in nm), None)
iapf_idx = next((i for i, nm in enumerate(all_feat_names) if "APF" in nm.upper() or "iapf" in nm.lower()), None)
smr_lat_idx = next((i for i, nm in enumerate(all_feat_names) if "CP" in nm and ("lat" in nm.lower() or "Lat" in nm)), None)

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

def scatter_panel(ax, x, y_acc, title, xlabel, rho, p):
    viable_mask = (y_acc >= 0.60)
    ax.scatter(x[viable_mask],  y_acc[viable_mask],  color=GREEN, alpha=0.7, s=35,
               edgecolors="none", label="Viable (≥60%)")
    ax.scatter(x[~viable_mask], y_acc[~viable_mask], color=RED,   alpha=0.7, s=35,
               edgecolors="none", label="Non-viable (<60%)")
    ax.axhline(0.60, color="gray", ls="--", lw=1, alpha=0.5)
    m, b = np.polyfit(x, y_acc, 1)
    xline = np.linspace(x.min(), x.max(), 100)
    ax.plot(xline, m*xline + b, color="black", lw=1.5, alpha=0.6)
    p_str = f"{p:.3f}" if p >= 0.001 else f"{p:.2e}"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fold-3 MI accuracy")
    ax.set_title(f"{title}\nSpearman ρ = {rho:+.3f},  p = {p_str}")
    ax.legend(fontsize=8, loc="upper left")

# Panel A: Spectral entropy
x_ent = X_all_aug[:, spec_ent_idx]
scatter_panel(axes[0,0], x_ent, fold3_acc,
              "A: Spectral entropy (EC)", "Spectral entropy (a.u.)",
              rho_vals[spec_ent_idx], p_vals[spec_ent_idx])

# Panel B: Parietal alpha at Pz
if pz_idx is not None:
    x_pz = X_all_aug[:, pz_idx]
    scatter_panel(axes[0,1], x_pz, fold3_acc,
                  "B: Parietal alpha power (Pz)", "log alpha power (dB)",
                  rho_vals[pz_idx], p_vals[pz_idx])
else:
    axes[0,1].text(0.5, 0.5, "Pz alpha not found in feature set",
                   ha="center", va="center", transform=axes[0,1].transAxes)

# Panel C: iAPF
if iapf_idx is not None:
    x_iapf = X_all_aug[:, iapf_idx]
    scatter_panel(axes[1,0], x_iapf, fold3_acc,
                  "C: Individual alpha peak frequency (iAPF)", "iAPF (Hz)",
                  rho_vals[iapf_idx], p_vals[iapf_idx])
else:
    # Use the SP feature with highest |rho|
    sorted_idx = np.argsort(np.abs(rho_vals))[::-1]
    alt_idx = sorted_idx[1] if sorted_idx[0] == spec_ent_idx else sorted_idx[0]
    x_alt = X_all_aug[:, alt_idx]
    scatter_panel(axes[1,0], x_alt, fold3_acc,
                  f"C: {all_feat_names[alt_idx] if alt_idx < len(all_feat_names) else 'Feature'}",
                  "Feature value",
                  rho_vals[alt_idx], p_vals[alt_idx])

# Panel D: CP3-CP4 SMR lateralisation
if smr_lat_idx is not None:
    x_lat = X_all_aug[:, smr_lat_idx]
    scatter_panel(axes[1,1], x_lat, fold3_acc,
                  "D: CP3-CP4 SMR lateralisation", "Lateralisation index",
                  rho_vals[smr_lat_idx], p_vals[smr_lat_idx])
else:
    # Use P4 alpha if available
    if p4_idx is not None:
        x_p4 = X_all_aug[:, p4_idx]
        scatter_panel(axes[1,1], x_p4, fold3_acc,
                      "D: Parietal alpha power (P4)", "log alpha power (dB)",
                      rho_vals[p4_idx], p_vals[p4_idx])

plt.tight_layout()
fig.savefig(FIGURES / "fig2_spearman_scatter.pdf", bbox_inches="tight")
fig.savefig(FIGURES / "fig2_spearman_scatter.png", bbox_inches="tight")
plt.close()
print("  Saved fig2_spearman_scatter", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: AUC comparison bar chart
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 3: AUC bar chart...", flush=True)

methods = [
    ("MS / LDA",                    0.286, RED,    "MS features"),
    ("MS / GB",                     0.513, RED,    "MS features"),
    ("EO / LR",                     0.477, ORANGE, "EO features"),
    ("SP / LDA",                    0.541, BLUE,   "SP features"),
    ("SP / LR",                     0.598, BLUE,   "SP features"),
    ("SP-47 / Nested-LR (k=20)",    0.625, GREEN,  "Nested selection"),
    ("SP-aug-68 / Nested-LR (k=28)",0.682, GREEN,  "Nested selection"),
    ("Clear subgroup (n=44)",        0.685, PURPLE, "Clear-label"),
]

labels = [m[0] for m in methods]
aucs   = [m[1] for m in methods]
colors = [m[2] for m in methods]

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.barh(labels, aucs, color=colors, alpha=0.85, edgecolor="white", height=0.65)
ax.axvline(0.5, color="gray", ls="--", lw=1.0, alpha=0.6, label="Chance (AUC=0.5)")
ax.axvline(0.74, color="gray", ls=":", lw=1.5, alpha=0.8, label="Label noise ceiling (≈0.74)")
for bar, auc in zip(bars, aucs):
    ax.text(auc + 0.005, bar.get_y() + bar.get_height()/2,
            f"{auc:.3f}", va="center", ha="left", fontsize=10, fontweight="bold")
ax.set_xlim(0.25, 0.78)
ax.set_xlabel("AUC-ROC")
ax.set_title("Classification performance across feature sets and classifiers")
ax.legend(fontsize=9)

# Add a legend for color coding
from matplotlib.patches import Patch
legend_els = [
    Patch(color=RED,    alpha=0.85, label="Microstate features"),
    Patch(color=ORANGE, alpha=0.85, label="EO spectral features"),
    Patch(color=BLUE,   alpha=0.85, label="Sensorimotor spectral (SP)"),
    Patch(color=GREEN,  alpha=0.85, label="Nested Spearman selection"),
    Patch(color=PURPLE, alpha=0.85, label="Clear-label subgroup"),
]
ax.legend(handles=legend_els, fontsize=9, loc="lower right")
plt.tight_layout()
fig.savefig(FIGURES / "fig3_auc_barplot.pdf", bbox_inches="tight")
fig.savefig(FIGURES / "fig3_auc_barplot.png", bbox_inches="tight")
plt.close()
print("  Saved fig3_auc_barplot", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: ROC curves for primary and secondary models
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 4: ROC curves (running LOSO for yt/yp)...", flush=True)

# Primary model: SP-aug-68/LR-nested-top28 (AUC=0.682)
print("  Running LOSO for SP-aug-68/LR-nested-top28...", flush=True)
yt_main, yp_main = loso_nested(X_aug, y, k=28, C=2.0)
auc_main = float(roc_auc_score(yt_main, yp_main))

# Standard model: SP-47/LR (all features, no selection)
print("  Running LOSO for SP-47/LR...", flush=True)
yt_sp, yp_sp = [], []
for i in range(n):
    tr = [j for j in range(n) if j != i]
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_sp[tr]); Xte = sc.transform(X_sp[[i]])
    lr  = LogisticRegression(solver="lbfgs", C=1.0, max_iter=2000)
    lr.fit(Xtr, y[tr])
    yt_sp.append(int(y[i])); yp_sp.append(float(lr.predict_proba(Xte)[0, 1]))
yt_sp, yp_sp = np.array(yt_sp), np.array(yp_sp)
auc_sp = float(roc_auc_score(yt_sp, yp_sp))

# Clear subgroup: SP-47/LR-nested-top10
borderline  = (fold3_acc >= 0.50) & (fold3_acc < 0.70)
clear_mask  = ~borderline
y_c = y[clear_mask]; X_sp_c = X_sp[clear_mask]; nc = len(y_c)
print(f"  Running LOSO for clear subgroup (n={nc})...", flush=True)
yt_cl, yp_cl = loso_nested(X_sp_c, y_c, k=10, C=2.0)
auc_cl = float(roc_auc_score(yt_cl, yp_cl))

fig, ax = plt.subplots(figsize=(7, 6))
for yt_, yp_, lbl, col, ls in [
    (yt_main, yp_main, f"SP-aug-68/LR-nested (full, n={n}), AUC={auc_main:.3f}", GREEN,  "-"),
    (yt_cl,   yp_cl,   f"SP-47/LR-nested (clear, n={nc}), AUC={auc_cl:.3f}", PURPLE, "--"),
    (yt_sp,   yp_sp,   f"SP-47/LR baseline (full, n={n}), AUC={auc_sp:.3f}", BLUE,   ":"),
]:
    fpr_, tpr_, _ = roc_curve(yt_, yp_)
    ax.plot(fpr_, tpr_, color=col, lw=2.0, ls=ls, label=lbl)

ax.plot([0,1], [0,1], "k--", lw=0.8, alpha=0.5, label="Chance")
ax.fill_between([0,1],[0,0],[0,0], alpha=0)  # spacer

# Mark Youden's J optimal point on primary model ROC
from sklearn.metrics import roc_curve as _roc
fpr_m, tpr_m, thresh_m = _roc(yt_main, yp_main)
j_idx   = np.argmax(tpr_m - fpr_m)
j_fpr   = fpr_m[j_idx]; j_tpr = tpr_m[j_idx]
ax.scatter([j_fpr], [j_tpr], marker="*", s=220, color=GREEN, zorder=5,
           edgecolors="black", linewidths=0.5,
           label=f"Youden's J: Sen={j_tpr:.2f}, Spe={1-j_fpr:.2f}")

ax.set_xlabel("False Positive Rate (1 - Specificity)")
ax.set_ylabel("True Positive Rate (Sensitivity)")
ax.set_title("ROC curves for key models\n(leave-one-subject-out cross-validation)")
ax.legend(fontsize=9, loc="lower right")
ax.axhline(0.74, color="gray", ls=":", lw=0.8, alpha=0.5)
ax.text(0.02, 0.95, "Label noise ceiling ≈ 0.74", fontsize=8, color="gray",
        transform=ax.transAxes)
plt.tight_layout()
fig.savefig(FIGURES / "fig4_roc_curves.pdf", bbox_inches="tight")
fig.savefig(FIGURES / "fig4_roc_curves.png", bbox_inches="tight")
plt.close()
print(f"  Saved fig4_roc_curves  [main={auc_main:.3f}, clear={auc_cl:.3f}]", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Feature importance (Spearman |rho| for top-20 features)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 5: Feature importance...", flush=True)

all_names_full = []
for nm in sp_names:
    all_names_full.append(nm)
for nm in extra_names:
    all_names_full.append(f"[ext] {nm}")
if len(all_names_full) < X_aug.shape[1]:
    all_names_full += [f"f{i}" for i in range(len(all_names_full), X_aug.shape[1])]

rho_abs = np.abs(rho_vals[:len(all_names_full)])
top20_idx = np.argsort(rho_abs)[::-1][:20]
top20_names = [all_names_full[i] if i < len(all_names_full) else f"f{i}"
               for i in top20_idx]
top20_rho  = [rho_vals[i] for i in top20_idx]
top20_p    = [p_vals[i]   for i in top20_idx]

# Colour by sign and significance
bar_colors = []
for rh, pv in zip(top20_rho, top20_p):
    if pv < 0.05:
        bar_colors.append(GREEN if rh > 0 else RED)
    else:
        bar_colors.append("#D1D5DB")

fig, ax = plt.subplots(figsize=(9, 7))
y_pos = np.arange(len(top20_names))
ax.barh(y_pos, [abs(r) for r in top20_rho], color=bar_colors, alpha=0.9)
ax.set_yticks(y_pos)
ax.set_yticklabels([nm.replace("BP_", "").replace("_", " ") for nm in top20_names[::-1]],
                    fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("|Spearman ρ| with Fold-3 accuracy")
ax.set_title("Top-20 predictive features (Spearman |ρ|, all n=92 subjects)")
ax.axvline(0.15, color="gray", ls="--", lw=0.8, alpha=0.5)

from matplotlib.patches import Patch
lels = [Patch(color=GREEN, label="p < 0.05, positive correlation"),
        Patch(color=RED,   label="p < 0.05, negative correlation"),
        Patch(color="#D1D5DB", label="p ≥ 0.05")]
ax.legend(handles=lels, fontsize=9)

# Annotate p-values
for k_, (rh, pv) in enumerate(zip(top20_rho, top20_p)):
    if pv < 0.05:
        ax.text(abs(rh) + 0.003, k_, f"p={pv:.3f}", va="center", fontsize=8)

plt.tight_layout()
fig.savefig(FIGURES / "fig5_feature_importance.pdf", bbox_inches="tight")
fig.savefig(FIGURES / "fig5_feature_importance.png", bbox_inches="tight")
plt.close()
print("  Saved fig5_feature_importance", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Microstate label-swap illustration
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 6: MS anti-prediction...", flush=True)

# Compare MS/LDA and MS/GB predictions
ms_means_by_class = []
for j in range(X_ms.shape[1]):
    mean_v  = X_ms[y==1, j].mean()
    mean_nv = X_ms[y==0, j].mean()
    ms_means_by_class.append((mean_v - mean_nv))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
ms_rhos = [spearmanr(X_ms[:, j], fold3_acc)[0] for j in range(X_ms.shape[1])]
ms_rho_mag = [abs(r) for r in ms_rhos]
ax.bar(range(len(ms_rhos)), ms_rhos,
       color=[RED if r < 0 else GREEN for r in ms_rhos], alpha=0.8)
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("Microstate feature index")
ax.set_ylabel("Spearman ρ with Fold-3 accuracy")
ax.set_title(f"A: MS feature-level Spearman ρ\n(max |ρ| = {max(ms_rho_mag):.3f})")
ax.set_ylim(-0.25, 0.25)

ax2 = axes[1]
# Bar chart showing LDA vs GB for MS
models = ["MS/LDA\n(linear)", "MS/LR\n(linear)", "MS/SVM\n(semi-lin.)",
          "MS/RF\n(non-linear)", "MS/GB\n(non-linear)"]
aucs_ms = [0.286, 0.294, 0.444, 0.392, 0.513]
bar_c = [RED, RED, ORANGE, GREEN, GREEN]
bars = ax2.bar(models, aucs_ms, color=bar_c, alpha=0.85, edgecolor="white")
ax2.axhline(0.5, color="gray", ls="--", lw=1.0, label="Chance")
ax2.text(0, 0.29, "Anti-predicts\n(1-AUC=0.714)", ha="center", fontsize=8,
         color="darkred", style="italic")
ax2.set_ylabel("AUC-ROC")
ax2.set_title("B: MS features: linear vs non-linear classifiers\n(MS/LDA anti-predicts; MS/GB captures non-linear signal)")
ax2.set_ylim(0.0, 0.65)
for bar, auc in zip(bars, aucs_ms):
    ax2.text(bar.get_x() + bar.get_width()/2, auc + 0.01,
             f"{auc:.3f}", ha="center", va="bottom", fontsize=9)
ax2.axhline(0.5, color="gray", ls="--", lw=1.0)

plt.tight_layout()
fig.savefig(FIGURES / "fig6_ms_antiprediction.pdf", bbox_inches="tight")
fig.savefig(FIGURES / "fig6_ms_antiprediction.png", bbox_inches="tight")
plt.close()
print("  Saved fig6_ms_antiprediction", flush=True)

print("\n" + "="*50)
print("All figures saved to:", FIGURES)
print(f"  fig1_label_distribution.pdf/png")
print(f"  fig2_spearman_scatter.pdf/png")
print(f"  fig3_auc_barplot.pdf/png")
print(f"  fig4_roc_curves.pdf/png    (main AUC={auc_main:.3f})")
print(f"  fig5_feature_importance.pdf/png")
print(f"  fig6_ms_antiprediction.pdf/png")
