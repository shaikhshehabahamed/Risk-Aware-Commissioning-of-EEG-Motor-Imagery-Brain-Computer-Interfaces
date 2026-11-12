"""
Script 04i: SP-aug-68 nested Spearman with optimal C=2.0 (from 04h).

04h confirmed: SP-47/LR(C=2.0)-top20 = 0.625
              Clear subgroup (n=44)/LR-top10 = 0.685

This script tests SP-aug-68 (47 SP + 21 extra spectral features) at C=2.0
to see if parietal alpha (Pz rho=+0.233, P4 rho=+0.231) pushes AUC higher.

Also: permutation test (n=200) for best model + bootstrap 95% CI.

Reads:  results/features.npz, results/features_extra.npz, results/viability_labels.json
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

# robust __file__ for both exec and file-based runs
try:
    _HERE = Path(__file__).resolve().parent
except NameError:
    _HERE = Path(r"C:\Users\sheha\Downloads\resting_bci_screening\scripts")

PROJ = _HERE.parent
RESULTS = PROJ / "results"
sys.path.insert(0, str(PROJ))

# ── Load ───────────────────────────────────────────────────────────────────────
data      = np.load(RESULTS / "features.npz", allow_pickle=True)
X_sp      = data["X_sp"].astype(float)

extra     = np.load(RESULTS / "features_extra.npz", allow_pickle=True)
X_extra   = extra["X_extra"].astype(float)
X_sp_aug  = np.hstack([X_sp, X_extra])   # 47 + 21 = 68 features

with open(RESULTS / "viability_labels.json") as f:
    ld = json.load(f)
y         = np.array([int(r["viable"])    for r in ld["per_subject"]])
fold3_acc = np.array([r["fold3_accuracy"] for r in ld["per_subject"]])
mean_acc  = np.array([r["mean_loro_acc"]  for r in ld["per_subject"]])
y_mean    = (mean_acc >= 0.60).astype(int)
n         = len(y)

print(f"n={n}  viable={y.sum()}  non-viable={(1-y).sum()}", flush=True)
print(f"SP-47 shape: {X_sp.shape}  SP-aug-68 shape: {X_sp_aug.shape}", flush=True)

C_OPT = 2.0    # optimal C from 04h


def loso_nested(X, y, k, C=C_OPT):
    """LOSO with within-fold Spearman feature selection + LR."""
    yt, yp = [], []
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if j != i]
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[[i]])
        rhos = np.array([abs(spearmanr(Xtr[:, j], y[tr])[0])
                         for j in range(Xtr.shape[1])])
        top  = np.argsort(rhos)[::-1][:k]
        lr   = LogisticRegression(solver="lbfgs", C=C, max_iter=2000)
        lr.fit(Xtr[:, top], y[tr])
        yt.append(int(y[i]))
        yp.append(float(lr.predict_proba(Xte[:, top])[0, 1]))
    return np.array(yt), np.array(yp)


# ── A. Sweep k on SP-aug-68 with C=2.0 ───────────────────────────────────────
print("\n" + "="*60)
print(f"A. SP-aug-68 nested Spearman (C={C_OPT}), full cohort")
print("="*60)
AUC_SP47_TOP20 = 0.625   # confirmed best from 04h

best_k_aug, best_auc_aug = 20, 0.0
best_yt_aug, best_yp_aug = None, None

for k in [15, 18, 20, 22, 25, 28, 30]:
    t0 = time.time()
    yt, yp = loso_nested(X_sp_aug, y, k=k, C=C_OPT)
    a = float(roc_auc_score(yt, yp))
    vs47 = ">" if a > AUC_SP47_TOP20 else "<="
    marker = " <-- BEST" if a > best_auc_aug else ""
    print(f"  SP-aug/LR(C={C_OPT})-top{k:2d}: AUC={a:.3f}  "
          f"({vs47} SP-47={AUC_SP47_TOP20})  [{time.time()-t0:.1f}s]{marker}", flush=True)
    if a > best_auc_aug:
        best_auc_aug, best_k_aug = a, k
        best_yt_aug, best_yp_aug = yt.copy(), yp.copy()

print(f"\n  SP-aug optimal: k={best_k_aug}  AUC={best_auc_aug:.3f}  "
      f"(vs SP-47=0.625: {'IMPROVEMENT' if best_auc_aug > 0.625 else 'no improvement'})",
      flush=True)


# ── B. Clear subgroup with SP-aug-68 ─────────────────────────────────────────
print("\n" + "="*60)
print("B. Clear subgroup (n=44) with SP-aug-68")
print("="*60)
borderline = (fold3_acc >= 0.50) & (fold3_acc < 0.70)
clear_mask  = ~borderline
y_c     = y[clear_mask];  X_aug_c = X_sp_aug[clear_mask]
nc      = len(y_c)
print(f"  n_clear={nc}  viable={y_c.sum()}  non-viable={(1-y_c).sum()}", flush=True)

AUC_CLEAR_SP47 = 0.685  # from 04h
clear_best_k, clear_best_auc = 10, 0.0
clear_best_yt, clear_best_yp = None, None
for k in [8, 10, 12, 15, 18]:
    t0 = time.time()
    yt_c, yp_c = loso_nested(X_aug_c, y_c, k=k, C=C_OPT)
    a_c = float(roc_auc_score(yt_c, yp_c))
    vs_ref = ">" if a_c > AUC_CLEAR_SP47 else "<="
    marker = " <-- BEST" if a_c > clear_best_auc else ""
    print(f"  Clear SP-aug/LR-top{k:2d}: AUC={a_c:.3f}  "
          f"({vs_ref} clear-SP-47=0.685)  [{time.time()-t0:.1f}s]{marker}", flush=True)
    if a_c > clear_best_auc:
        clear_best_auc, clear_best_k = a_c, k
        clear_best_yt, clear_best_yp = yt_c.copy(), yp_c.copy()

print(f"\n  Clear optimal: k={clear_best_k}  AUC={clear_best_auc:.3f}", flush=True)


# ── C. Mean-LORO label with SP-aug-68 ────────────────────────────────────────
print("\n" + "="*60)
print("C. Mean-LORO label with SP-aug-68")
print("="*60)
mean_best_k, mean_best_auc = 20, 0.0
mean_best_yt, mean_best_yp = None, None
for k in [18, 20, 22, 25, 28, 30]:
    t0 = time.time()
    yt_m, yp_m = loso_nested(X_sp_aug, y_mean, k=k, C=C_OPT)
    a_m = float(roc_auc_score(yt_m, yp_m))
    marker = " <-- BEST" if a_m > mean_best_auc else ""
    print(f"  MeanLabel SP-aug/LR-top{k:2d}: AUC={a_m:.3f}  "
          f"[{time.time()-t0:.1f}s]{marker}", flush=True)
    if a_m > mean_best_auc:
        mean_best_auc, mean_best_k = a_m, k
        mean_best_yt, mean_best_yp = yt_m.copy(), yp_m.copy()

print(f"\n  Mean-label optimal: k={mean_best_k}  AUC={mean_best_auc:.3f}", flush=True)


# ── D. Permutation test for primary best model ────────────────────────────────
# Primary best = whichever is highest: SP-47/LR(C=2)-top20 vs SP-aug-68/LR(C=2)-topK
if best_auc_aug > AUC_SP47_TOP20:
    X_perm = X_sp_aug;  k_perm = best_k_aug;  obs_auc = best_auc_aug
    label_perm = "SP-aug-68"
else:
    X_perm = X_sp;  k_perm = 20;  obs_auc = AUC_SP47_TOP20
    label_perm = "SP-47"

print("\n" + "="*60)
print(f"D. Permutation test (n=200) for {label_perm}/LR(C={C_OPT})-top{k_perm}")
print(f"   Observed AUC = {obs_auc:.3f}")
print("="*60)
rng = np.random.default_rng(42)
N_PERM = 200
perm_aucs = []
t0 = time.time()
for pi in range(N_PERM):
    y_perm = rng.permutation(y)
    yt_p, yp_p = loso_nested(X_perm, y_perm, k=k_perm, C=C_OPT)
    perm_aucs.append(float(roc_auc_score(yt_p, yp_p)))
    if (pi+1) % 20 == 0:
        elapsed = time.time()-t0
        eta     = elapsed/(pi+1) * (N_PERM-pi-1)
        print(f"  perm {pi+1:3d}/{N_PERM}  max_null={max(perm_aucs):.3f}  "
              f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

perm_aucs = np.array(perm_aucs)
p_val = float((perm_aucs >= obs_auc).mean())
sig   = "p<0.005" if p_val == 0.0 else f"p={p_val:.3f}"
print(f"\n  Observed AUC = {obs_auc:.3f}")
print(f"  Null mean    = {perm_aucs.mean():.3f} +/- {perm_aucs.std():.3f}")
print(f"  Null max     = {perm_aucs.max():.3f}")
print(f"  p-value      = {p_val:.4f}  ({sig})", flush=True)
sig_label = "SIGNIFICANT (p<0.05)" if p_val < 0.05 else "NOT significant"
print(f"  --> {sig_label}")


# ── E. Bootstrap 95% CI for best models ──────────────────────────────────────
print("\n" + "="*60)
print("E. Bootstrap 95% CI (n=5000)")
print("="*60)
rng2 = np.random.default_rng(99)
N_BOOT = 5000

def bootstrap_ci(yt, yp, n_boot, rng, label):
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(yt), len(yt))
        yt_b, yp_b = yt[idx], yp[idx]
        if len(np.unique(yt_b)) < 2: continue
        boot.append(float(roc_auc_score(yt_b, yp_b)))
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    obs = float(roc_auc_score(yt, yp))
    print(f"  {label}: AUC={obs:.3f}  95%CI=[{lo:.3f},{hi:.3f}]", flush=True)
    return obs, lo, hi

# Use stored best_yt/yp from LOSO runs
if best_auc_aug > AUC_SP47_TOP20:
    main_yt, main_yp = best_yt_aug, best_yp_aug
    main_label = f"SP-aug-68/LR(C={C_OPT})-top{best_k_aug}"
else:
    # Re-run SP-47 at k=20, C=2.0 to get stored yt/yp
    print("  Re-running SP-47/LR(C=2.0)-top20 to get yt/yp...", flush=True)
    main_yt, main_yp = loso_nested(X_sp, y, k=20, C=2.0)
    main_label = "SP-47/LR(C=2.0)-top20"

ci_main = bootstrap_ci(main_yt, main_yp, N_BOOT, rng2, main_label)
ci_clear = bootstrap_ci(clear_best_yt, clear_best_yp, N_BOOT, rng2,
                        f"Clear subgroup (n={nc}) SP-aug/LR-top{clear_best_k}")
ci_mean  = bootstrap_ci(mean_best_yt, mean_best_yp, N_BOOT, rng2,
                        f"MeanLabel SP-aug/LR-top{mean_best_k}")


# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"  Best model (full cohort):     {main_label}")
print(f"    AUC = {ci_main[0]:.3f}  95%CI=[{ci_main[1]:.3f},{ci_main[2]:.3f}]")
print(f"    Permutation p = {p_val:.4f}  ({label_perm}/LR(C={C_OPT})-top{k_perm})")
print(f"  Clear subgroup (n={nc}):         SP-aug/LR-top{clear_best_k}")
print(f"    AUC = {ci_clear[0]:.3f}  95%CI=[{ci_clear[1]:.3f},{ci_clear[2]:.3f}]")
print(f"  Mean-LORO label:              SP-aug/LR-top{mean_best_k}")
print(f"    AUC = {ci_mean[0]:.3f}  95%CI=[{ci_mean[1]:.3f},{ci_mean[2]:.3f}]")

results = {
    "sp_aug_best_k": best_k_aug, "sp_aug_best_auc": best_auc_aug,
    "sp47_auc": AUC_SP47_TOP20,
    "best_overall": max(best_auc_aug, AUC_SP47_TOP20),
    "ci_main_lo": ci_main[1], "ci_main_hi": ci_main[2],
    "perm_feature_set": label_perm, "perm_k": k_perm,
    "perm_p": p_val, "perm_null_mean": float(perm_aucs.mean()),
    "clear_subgroup_n": int(nc),
    "clear_subgroup_auc": clear_best_auc,
    "clear_subgroup_k": clear_best_k,
    "ci_clear_lo": ci_clear[1], "ci_clear_hi": ci_clear[2],
    "mean_label_auc": mean_best_auc, "mean_label_k": mean_best_k,
    "ci_mean_lo": ci_mean[1], "ci_mean_hi": ci_mean[2],
}
with open(RESULTS / "screening_results_best_model.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {RESULTS / 'screening_results_best_model.json'}", flush=True)
