"""
Script 04h: Fine-tune nested Spearman feature selection (optimal k).

Finding from 04d:
  SP/LR-top5=0.534, top10=0.517, top15=0.586, top20=0.623, all-47=0.598
  Optimum near k=20. Test k=18,20,22,25,28,30,35 and clear subgroup.

Also tests:
  - LR with C tuning (C=0.01, 0.1, 0.5, 1, 2, 5) at optimal k
  - Mean LORO label vs Fold-3 label at optimal k

Reads:  results/features.npz, results/viability_labels.json
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RESULTS

data     = np.load(RESULTS / "features.npz", allow_pickle=True)
subjects = list(data["subjects"])
X_ms     = data["X_ms"].astype(float)
X_sp     = data["X_sp"].astype(float)
X_eo     = data["X_eo"].astype(float)
X_full   = np.hstack([X_ms, X_sp, X_eo])

with open(RESULTS / "viability_labels.json") as f:
    ld = json.load(f)
y          = np.array([int(r["viable"])        for r in ld["per_subject"]])
fold3_acc  = np.array([r["fold3_accuracy"]     for r in ld["per_subject"]])
mean_acc   = np.array([r["mean_loro_acc"]      for r in ld["per_subject"]])
n          = len(y)
y_mean     = (mean_acc >= 0.60).astype(int)

print(f"n={n}  viable={y.sum()}  non-viable={(1-y).sum()}")
print(f"Mean-label: viable={y_mean.sum()}  non-viable={(1-y_mean).sum()}")


def loso_spearman_lr(X, y, k, C=1.0):
    """LOSO with nested Spearman top-k selection + LR(C)."""
    yt, yp = [], []
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if j != i]
        Xtr, ytr = X[tr], y[tr]
        Xte, yte = X[[i]], y[[i]]
        sc  = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)
        rhos = np.array([abs(spearmanr(Xtr_s[:, j], ytr)[0]) for j in range(Xtr_s.shape[1])])
        top  = np.argsort(rhos)[::-1][:k]
        lr   = LogisticRegression(solver="lbfgs", C=C, max_iter=2000)
        lr.fit(Xtr_s[:, top], ytr)
        yt.append(int(yte[0]))
        yp.append(float(lr.predict_proba(Xte_s[:, top])[0, 1]))
    yt, yp = np.array(yt), np.array(yp)
    return float(roc_auc_score(yt, yp)) if len(np.unique(yt)) == 2 else float("nan"), yt, yp


# ── A. Sweep k (full cohort, y=fold3) ─────────────────────────────────────────
print("\n" + "="*60)
print("A. Sweep k for SP, full cohort, y=fold3")
print("="*60)
best_k, best_auc = 20, 0.0
for k in [16, 18, 20, 22, 24, 26, 28, 30, 35, 40]:
    t0 = time.time()
    a, yt, yp = loso_spearman_lr(X_sp, y, k=k)
    marker = " <-- BEST" if a > best_auc else ""
    print(f"  SP/LR-top{k:2d}: AUC={a:.3f}  [{time.time()-t0:.1f}s]{marker}")
    if a > best_auc:
        best_auc, best_k = a, k
        best_yt, best_yp = yt, yp

print(f"\n  Optimal: k={best_k}  AUC={best_auc:.3f}")


# ── B. Sweep C at optimal k ────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"B. Sweep C at k={best_k}, full cohort, y=fold3")
print("="*60)
best_C, best_auc_C = 1.0, 0.0
for C in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
    t0 = time.time()
    a, yt_c, yp_c = loso_spearman_lr(X_sp, y, k=best_k, C=C)
    marker = " <-- BEST" if a > best_auc_C else ""
    print(f"  SP/LR(C={C:5.2f})-top{best_k}: AUC={a:.3f}  [{time.time()-t0:.1f}s]{marker}")
    if a > best_auc_C:
        best_auc_C, best_C = a, C

print(f"\n  Optimal: k={best_k}  C={best_C}  AUC={best_auc_C:.3f}")


# ── C. Clear subgroup at best k and C ─────────────────────────────────────────
print("\n" + "="*60)
print(f"C. Clear subgroup (n=44), k=best, C=best")
print("="*60)
borderline = (fold3_acc >= 0.50) & (fold3_acc < 0.70)
clear_mask  = ~borderline
y_c         = y[clear_mask]
X_sp_c      = X_sp[clear_mask]
nc          = len(y_c)
print(f"  n_clear={nc}  viable={y_c.sum()}  non-viable={(1-y_c).sum()}")

clear_best_k, clear_best_auc = best_k, 0.0
for k in [10, 15, 18, best_k, 22, 25, 30]:
    t0 = time.time()
    a, _, _ = loso_spearman_lr(X_sp_c, y_c, k=k, C=best_C)
    marker = " <-- BEST" if a > clear_best_auc else ""
    print(f"  Clear-SP/LR-top{k}: AUC={a:.3f}  [{time.time()-t0:.1f}s]{marker}")
    if a > clear_best_auc:
        clear_best_auc, clear_best_k = a, k

print(f"\n  Clear optimal: k={clear_best_k}  AUC={clear_best_auc:.3f}")


# ── D. Mean LORO label ────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"D. Mean LORO label, k=best, C=best")
print("="*60)
a_mean_fold3, _, _ = loso_spearman_lr(X_sp, y_mean, k=best_k, C=best_C)
print(f"  SP/LR-top{best_k} (mean label): AUC={a_mean_fold3:.3f}")

# Also sweep k for mean label
print("  Sweep k (mean label):")
mean_best_k, mean_best_auc = best_k, 0.0
for k in [15, 18, 20, 22, 25, 30]:
    a_m, _, _ = loso_spearman_lr(X_sp, y_mean, k=k, C=best_C)
    marker = " <-- BEST" if a_m > mean_best_auc else ""
    print(f"    k={k:2d}: AUC={a_m:.3f}{marker}")
    if a_m > mean_best_auc:
        mean_best_auc, mean_best_k = a_m, k


# ── E. EO + SP combined (with Spearman selection) ─────────────────────────────
print("\n" + "="*60)
print(f"E. SP+EO combined with Spearman selection, k=best")
print("="*60)
X_speo = np.hstack([X_sp, X_eo])  # 47+42=89
for k in [20, 25, 30, 35]:
    t0 = time.time()
    a, _, _ = loso_spearman_lr(X_speo, y, k=k, C=best_C)
    print(f"  SP+EO/LR-top{k}: AUC={a:.3f}  [{time.time()-t0:.1f}s]")


# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("OPTIMIZATION SUMMARY")
print("="*60)
print(f"  Optimal k for SP features:    {best_k}")
print(f"  Optimal C for LR:             {best_C}")
print(f"  Best AUC (full cohort):       {best_auc_C:.3f}")
print(f"  Best AUC (clear subgroup):    {clear_best_auc:.3f}")
print(f"  Best AUC (mean label):        {mean_best_auc:.3f}")

results = {
    "best_k": best_k, "best_C": best_C,
    "full_cohort_auc": best_auc_C,
    "clear_subgroup_auc": clear_best_auc,
    "mean_label_auc": mean_best_auc,
}
import json
with open(RESULTS / "screening_results_optimized_k.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {RESULTS / 'screening_results_optimized_k.json'}")
