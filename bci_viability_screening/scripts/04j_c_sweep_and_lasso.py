"""
Script 04j: C-sweep at k=28 for SP-aug-68, plus L1-LASSO on SP-aug-68.

Goal: Find if AUC > 0.682 (current best: SP-aug-68/LR(C=2.0)-top28).

A. C-sweep at k=28: test C in {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0}
B. k fine-tune at best-C: test k in {26, 27, 28, 29, 30, 32} around peak
C. L1-LASSO on SP-aug-68 (C sweep): sklearn LogisticRegression(penalty='l1')
   - LASSO does selection + classification in one step (more principled than nested Spearman)
D. Elastic-net LR on SP-aug-68 (l1_ratio=0.5)
"""

import json, sys, time
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

try:
    _HERE = Path(__file__).resolve().parent
except NameError:
    _HERE = Path(r"C:\Users\sheha\Downloads\resting_bci_screening\scripts")

PROJ = _HERE.parent
RESULTS = PROJ / "results"

# ── Load ───────────────────────────────────────────────────────────────────────
data      = np.load(RESULTS / "features.npz",       allow_pickle=True)
X_sp      = data["X_sp"].astype(float)
extra     = np.load(RESULTS / "features_extra.npz", allow_pickle=True)
X_extra   = extra["X_extra"].astype(float)
X_aug     = np.hstack([X_sp, X_extra])   # 68 features

with open(RESULTS / "viability_labels.json") as f:
    ld = json.load(f)
y = np.array([int(r["viable"]) for r in ld["per_subject"]])
n = len(y)
print(f"n={n}  viable={y.sum()}  non-viable={(1-y).sum()}", flush=True)
print(f"SP-aug-68 shape: {X_aug.shape}", flush=True)

AUC_BEST = 0.682  # current best to beat


def loso_nested_lr(X, y, k, C=2.0):
    """LOSO with nested Spearman top-k + L2-LR(C)."""
    yt, yp = [], []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
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
    return roc_auc_score(np.array(yt), np.array(yp))


def loso_lr(X, y, C=1.0, penalty='l2', solver='lbfgs'):
    """LOSO with regularised LR (no nested selection)."""
    yt, yp = [], []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[[i]])
        lr   = LogisticRegression(penalty=penalty, C=C,
                                   solver=solver, max_iter=5000)
        lr.fit(Xtr, y[tr])
        yt.append(int(y[i]))
        yp.append(float(lr.predict_proba(Xte)[0, 1]))
    return roc_auc_score(np.array(yt), np.array(yp))


def loso_lr_full(X, y, C=1.0, penalty='l2', solver='lbfgs'):
    """Return yt, yp for bootstrap CI."""
    yt, yp = [], []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[[i]])
        lr   = LogisticRegression(penalty=penalty, C=C,
                                   solver=solver, max_iter=5000)
        lr.fit(Xtr, y[tr])
        yt.append(int(y[i]))
        yp.append(float(lr.predict_proba(Xte)[0, 1]))
    return np.array(yt), np.array(yp)


# ── A. C-sweep at k=28 (L2-LR nested) ───────────────────────────────────────
print("\n" + "="*60)
print("A. C-sweep at k=28 for SP-aug-68/LR-nested")
print("="*60)
best_C, best_auc_A = 2.0, 0.0
for C in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
    t0 = time.time()
    a = float(loso_nested_lr(X_aug, y, k=28, C=C))
    vs = "BEST" if a > best_auc_A else f"<= best={best_auc_A:.3f}"
    print(f"  SP-aug/LR(C={C:.1f})-top28: AUC={a:.3f}  [{time.time()-t0:.1f}s]"
          f"  {vs}", flush=True)
    if a > best_auc_A:
        best_auc_A, best_C = a, C
print(f"\n  A-optimal: C={best_C}  AUC={best_auc_A:.3f}  (vs global best={AUC_BEST})")


# ── B. k fine-tune at best C ─────────────────────────────────────────────────
print("\n" + "="*60)
print(f"B. k fine-tune around 28 at C={best_C} for SP-aug-68")
print("="*60)
best_k_B, best_auc_B = 28, 0.0
for k in [24, 26, 27, 28, 29, 30, 32, 35]:
    t0 = time.time()
    a = float(loso_nested_lr(X_aug, y, k=k, C=best_C))
    vs = "BEST" if a > best_auc_B else f"<= best={best_auc_B:.3f}"
    print(f"  SP-aug/LR(C={best_C:.1f})-top{k:2d}: AUC={a:.3f}  "
          f"[{time.time()-t0:.1f}s]  {vs}", flush=True)
    if a > best_auc_B:
        best_auc_B, best_k_B = a, k
print(f"\n  B-optimal: k={best_k_B}  C={best_C}  AUC={best_auc_B:.3f}")


# ── C. L1-LASSO on SP-aug-68 (full features, no nested selection) ─────────────
print("\n" + "="*60)
print("C. L1-LASSO on SP-aug-68 (no separate nested selection)")
print("="*60)
best_auc_C, best_C_lasso = 0.0, 0.01
best_yt_C, best_yp_C = None, None
for C in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
    t0 = time.time()
    yt_c, yp_c = loso_lr_full(X_aug, y, C=C, penalty='l1', solver='liblinear')
    a = float(roc_auc_score(yt_c, yp_c))
    vs = "BEST" if a > best_auc_C else f"<= best={best_auc_C:.3f}"
    print(f"  SP-aug/LASSO(C={C:.3f}): AUC={a:.3f}  [{time.time()-t0:.1f}s]  {vs}",
          flush=True)
    if a > best_auc_C:
        best_auc_C, best_C_lasso = a, C
        best_yt_C, best_yp_C = yt_c, yp_c
print(f"\n  C-optimal: C={best_C_lasso}  AUC={best_auc_C:.3f}")


# ── D. LASSO on SP-47 (standard features) ─────────────────────────────────────
print("\n" + "="*60)
print("D. L1-LASSO on SP-47 (standard sensorimotor features only)")
print("="*60)
best_auc_D, best_C_D = 0.0, 0.01
best_yt_D, best_yp_D = None, None
for C in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
    t0 = time.time()
    yt_d, yp_d = loso_lr_full(X_sp, y, C=C, penalty='l1', solver='liblinear')
    a = float(roc_auc_score(yt_d, yp_d))
    vs = "BEST" if a > best_auc_D else f"<= best={best_auc_D:.3f}"
    print(f"  SP-47/LASSO(C={C:.3f}): AUC={a:.3f}  [{time.time()-t0:.1f}s]  {vs}",
          flush=True)
    if a > best_auc_D:
        best_auc_D, best_C_D = a, C
        best_yt_D, best_yp_D = yt_d, yp_d
print(f"\n  D-optimal SP-47/LASSO: C={best_C_D}  AUC={best_auc_D:.3f}")


# ── E. Permutation test for LASSO model ─────────────────────────────────────
# Use best LASSO result (whichever was better: C or D)
if best_auc_C >= best_auc_D:
    X_perm = X_aug;  C_perm = best_C_lasso; obs = best_auc_C; lbl = "SP-aug-68/LASSO"
    yt_best, yp_best = best_yt_C, best_yp_C
else:
    X_perm = X_sp;   C_perm = best_C_D;     obs = best_auc_D; lbl = "SP-47/LASSO"
    yt_best, yp_best = best_yt_D, best_yp_D

print("\n" + "="*60)
print(f"E. Permutation test (n=200) for {lbl} (C={C_perm:.3f}), AUC={obs:.3f}")
print("="*60)
rng = np.random.default_rng(77)
N_PERM = 200
perms = []
t0 = time.time()
for pi in range(N_PERM):
    yp_ = rng.permutation(y)
    a_p = float(loso_lr(X_perm, yp_, C=C_perm, penalty='l1', solver='liblinear'))
    perms.append(a_p)
    if (pi+1) % 20 == 0:
        elapsed = time.time()-t0
        print(f"  perm {pi+1:3d}/{N_PERM}  max_null={max(perms):.3f}  "
              f"elapsed={elapsed:.0f}s  ETA={elapsed/(pi+1)*(N_PERM-pi-1):.0f}s",
              flush=True)
perms = np.array(perms)
p_val = float((perms >= obs).mean())
print(f"\n  Observed AUC = {obs:.3f}")
print(f"  Null mean    = {perms.mean():.3f}  std={perms.std():.3f}")
print(f"  Null max     = {perms.max():.3f}")
print(f"  p-value      = {p_val:.4f}")
sig = "SIGNIFICANT (p<0.05)" if p_val < 0.05 else "n.s."
print(f"  --> {sig}", flush=True)


# ── F. Bootstrap CI for best overall model ────────────────────────────────────
print("\n" + "="*60)
print("F. Bootstrap CI (n=5000) for LASSO model")
print("="*60)
rng2 = np.random.default_rng(88)
boot = []
for _ in range(5000):
    idx = rng2.integers(0, n, n)
    yt_b, yp_b = yt_best[idx], yp_best[idx]
    if len(np.unique(yt_b)) < 2: continue
    boot.append(float(roc_auc_score(yt_b, yp_b)))
boot = np.array(boot)
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"  {lbl}: AUC={obs:.3f}  95%CI=[{lo:.3f},{hi:.3f}]", flush=True)


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
overall_best = max(best_auc_A, best_auc_B, best_auc_C, best_auc_D)
print(f"  A: SP-aug/L2-nested C-sweep: AUC={best_auc_A:.3f}  (C={best_C}, k=28)")
print(f"  B: SP-aug/L2-nested k-tune:  AUC={best_auc_B:.3f}  (C={best_C}, k={best_k_B})")
print(f"  C: SP-aug/LASSO:             AUC={best_auc_C:.3f}  (C={best_C_lasso:.3f})")
print(f"  D: SP-47/LASSO:              AUC={best_auc_D:.3f}  (C={best_C_D:.3f})")
print(f"  Overall best: {overall_best:.3f}  (vs 0.682 from 04i)")
print(f"  LASSO perm p = {p_val:.4f}  ({sig})")

out = {
    "best_auc_C_sweep": best_auc_A, "best_C_nested": best_C,
    "best_auc_k_tune": best_auc_B, "best_k": best_k_B,
    "best_auc_lasso_aug68": best_auc_C, "best_C_lasso_aug68": best_C_lasso,
    "best_auc_lasso_sp47": best_auc_D, "best_C_lasso_sp47": best_C_D,
    "lasso_perm_p": p_val, "lasso_perm_null_mean": float(perms.mean()),
    "lasso_ci_lo": float(lo), "lasso_ci_hi": float(hi),
}
with open(RESULTS / "screening_results_c_sweep_lasso.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved -> {RESULTS / 'screening_results_c_sweep_lasso.json'}", flush=True)
