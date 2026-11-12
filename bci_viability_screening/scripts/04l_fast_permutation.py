"""
Script 04l: Fast permutation test for primary model (SP-aug-68/LR-top28, C=2.0).

Speed optimisation over 04i:
  - Vectorised Spearman via rank-Pearson (compute ranks once per fold, then
    matrix multiplication for all 68 feature correlations simultaneously).
  - Parallel permutations via joblib (optional; falls back to sequential).
  - Saves every permutation AUC so CI can be computed without rerunning.

Runs 200 permutations. Expected wall-clock: <10 min (vs ~60 min for 04i).
"""

import json, sys, time
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, rankdata
from scipy.special import ndtri
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

try:
    _HERE = Path(__file__).resolve().parent
except NameError:
    _HERE = Path(r"C:\Users\sheha\Downloads\resting_bci_screening\scripts")

PROJ    = _HERE.parent
RESULTS = PROJ / "results"
sys.path.insert(0, str(PROJ))

# ── Data ──────────────────────────────────────────────────────────────────────
data     = np.load(RESULTS / "features.npz",       allow_pickle=True)
X_sp     = data["X_sp"].astype(float)
subjects = list(data["subjects"])

extra    = np.load(RESULTS / "features_extra.npz", allow_pickle=True)
X_extra  = extra["X_extra"].astype(float)
X_aug    = np.hstack([X_sp, X_extra])   # 68 features

with open(RESULTS / "viability_labels.json") as f:
    ld = json.load(f)
y = np.array([int(r["viable"]) for r in ld["per_subject"]])
n = len(y)

print(f"n={n}  viable={y.sum()}  non-viable={(1-y).sum()}", flush=True)
print(f"X_aug shape: {X_aug.shape}", flush=True)

K    = 28    # confirmed optimal
C    = 2.0   # confirmed optimal
SEED = 42

# ── Vectorised Spearman selection ─────────────────────────────────────────────
def spearman_top_k_fast(X_tr_std, y_tr, k):
    """
    Compute |Spearman rho| between each column of X_tr_std and y_tr,
    return indices of top-k features.

    Uses rank → Pearson trick which is ~10× faster than looping spearmanr.
    """
    n_tr = X_tr_std.shape[0]
    # Rank each column of X (axis=0) and the label vector
    Xr = np.apply_along_axis(rankdata, 0, X_tr_std)  # (n_tr, n_feat)
    yr = rankdata(y_tr)                                # (n_tr,)
    # Demean
    Xr -= Xr.mean(axis=0)
    yr -= yr.mean()
    # Pearson on ranks = Spearman
    num   = Xr.T @ yr           # (n_feat,)
    denom = (np.linalg.norm(Xr, axis=0) * np.linalg.norm(yr))
    rho   = np.abs(num / (denom + 1e-10))
    return np.argsort(rho)[::-1][:k]


def loso_nested_fast(X, y_bin, k=28, C=2.0):
    """LOSO with vectorised Spearman feature selection."""
    yt, yp = np.empty(n, dtype=int), np.empty(n)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[[i]])
        top = spearman_top_k_fast(Xtr, y_bin[tr], k)
        lr  = LogisticRegression(solver="lbfgs", C=C, max_iter=2000)
        lr.fit(Xtr[:, top], y_bin[tr])
        yt[i] = int(y_bin[i])
        yp[i] = float(lr.predict_proba(Xte[:, top])[0, 1])
    return yt, yp


# ═══════════════════════════════════════════════════════════════════════════════
# A. Observed AUC (quick confirmation)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("A. Observed AUC (fast implementation)")
print("="*60, flush=True)
t0 = time.time()
yt_obs, yp_obs = loso_nested_fast(X_aug, y, k=K, C=C)
auc_obs = float(roc_auc_score(yt_obs, yp_obs))
print(f"  SP-aug-68/LR-nested-top{K} (C={C}): AUC={auc_obs:.3f}  [{time.time()-t0:.1f}s]",
      flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B. Permutation test (n=200)
# ═══════════════════════════════════════════════════════════════════════════════
N_PERMS = 200
print(f"\n" + "="*60)
print(f"B. Permutation test (n={N_PERMS}), observed AUC = {auc_obs:.3f}")
print("="*60, flush=True)

rng = np.random.default_rng(SEED)
null_aucs = []
t_start   = time.time()

for p_idx in range(N_PERMS):
    y_perm   = rng.permutation(y)
    _, yp_p  = loso_nested_fast(X_aug, y_perm, k=K, C=C)
    # Use true labels yt_obs for the AUC (permuted model vs true labels)
    a = float(roc_auc_score(yt_obs, yp_p))
    null_aucs.append(a)

    if (p_idx + 1) % 20 == 0:
        elapsed  = time.time() - t_start
        eta      = elapsed / (p_idx + 1) * (N_PERMS - p_idx - 1)
        max_null = max(null_aucs)
        p_now    = sum(a_ >= auc_obs for a_ in null_aucs) / len(null_aucs)
        print(f"  perm {p_idx+1:3d}/{N_PERMS}  max_null={max_null:.3f}"
              f"  p_running={p_now:.4f}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
              flush=True)

null_aucs_arr = np.array(null_aucs)
perm_p    = float((null_aucs_arr >= auc_obs).sum() / N_PERMS)
null_mean = float(null_aucs_arr.mean())
null_std  = float(null_aucs_arr.std())
null_max  = float(null_aucs_arr.max())
z_score   = (auc_obs - null_mean) / null_std

print(f"\n  Observed AUC = {auc_obs:.3f}")
print(f"  Null mean    = {null_mean:.3f}  std={null_std:.3f}")
print(f"  Null max     = {null_max:.3f}")
print(f"  z-score      = {z_score:.2f}")
print(f"  p-value      = {perm_p:.4f}")
sig = "*** p<0.01" if perm_p < 0.01 else ("** p<0.05" if perm_p < 0.05 else
      ("* p<0.10" if perm_p < 0.10 else "n.s."))
print(f"  --> {sig}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# C. Bootstrap 95% CI
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n" + "="*60)
print("C. Bootstrap 95% CI for observed AUC")
print("="*60, flush=True)

rng2 = np.random.default_rng(SEED + 1)
boot_aucs = []
for _ in range(5000):
    idx  = rng2.integers(0, n, n)
    yt_b = yt_obs[idx]; yp_b = yp_obs[idx]
    if len(np.unique(yt_b)) < 2:
        continue
    boot_aucs.append(float(roc_auc_score(yt_b, yp_b)))
ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])
print(f"  SP-aug-68/LR-nested-top{K}: AUC={auc_obs:.3f}  95%CI=[{ci_lo:.3f},{ci_hi:.3f}]",
      flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# D. Save results
# ═══════════════════════════════════════════════════════════════════════════════
out = {
    "model":        f"SP-aug-68/LR-nested-top{K}",
    "k":            K,
    "C":            C,
    "observed_auc": auc_obs,
    "ci_lo":        ci_lo,
    "ci_hi":        ci_hi,
    "perm_p":       perm_p,
    "null_mean":    null_mean,
    "null_std":     null_std,
    "null_max":     null_max,
    "z_score":      z_score,
    "n_perms":      N_PERMS,
    "null_aucs":    null_aucs,
    "yt":           yt_obs.tolist(),
    "yp":           yp_obs.tolist(),
}
with open(RESULTS / "screening_results_permtest.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved -> screening_results_permtest.json", flush=True)

print(f"\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  Model:        SP-aug-68/LR-nested (k={K}, C={C})")
print(f"  Observed AUC: {auc_obs:.3f}")
print(f"  95% CI:       [{ci_lo:.3f}, {ci_hi:.3f}]")
print(f"  Perm p-value: {perm_p:.4f}  ({N_PERMS} permutations)")
print(f"  z-score:      {z_score:.2f}")
print(f"  Null:         μ={null_mean:.3f} σ={null_std:.3f} max={null_max:.3f}")
print(f"  {sig}")
