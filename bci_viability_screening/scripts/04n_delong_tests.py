"""
Script 04n: DeLong paired AUC tests for Table 4.

Runs LOSO for MS/LDA, SP/LDA, EO/LDA, MS+SP/LDA, FULL/LDA to get
per-subject prediction vectors, then performs DeLong paired AUC tests
for the 4 planned comparisons.

DeLong AUC variance estimator (DeLong et al. 1988) is computed from
structural components of the two correlated AUC estimates.
"""

import json, sys, time
import numpy as np
from pathlib import Path
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

try:
    _HERE = Path(__file__).resolve().parent
except NameError:
    _HERE = Path(r"C:\Users\sheha\Downloads\resting_bci_screening\scripts")

PROJ    = _HERE.parent
RESULTS = PROJ / "results"
sys.path.insert(0, str(PROJ))

# ── Data ──────────────────────────────────────────────────────────────────────
data     = np.load(RESULTS / "features.npz", allow_pickle=True)
X_sp     = data["X_sp"].astype(float)
X_ms     = data["X_ms"].astype(float)
X_eo     = data["X_eo"].astype(float)

with open(RESULTS / "viability_labels.json") as f:
    ld = json.load(f)
y = np.array([int(r["viable"]) for r in ld["per_subject"]])
n = len(y)

print(f"n={n}  viable={y.sum()}  non-viable={(1-y).sum()}", flush=True)

X_mssp = np.hstack([X_ms, X_sp])   # 73
X_full = np.hstack([X_ms, X_sp, X_eo])  # 115


# ── LOSO helper ───────────────────────────────────────────────────────────────
def loso_lda(X, y_bin):
    yt, yp = [], []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[[i]])
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(Xtr, y_bin[tr])
        yt.append(int(y_bin[i]))
        yp.append(float(lda.predict_proba(Xte)[0, 1]))
    return np.array(yt), np.array(yp)


# ── DeLong AUC test ───────────────────────────────────────────────────────────
def structural_components(y_true, y_pred):
    """Compute structural components V10 and V01 (DeLong 1988 kernel method)."""
    pos = y_pred[y_true == 1]
    neg = y_pred[y_true == 0]
    n1, n0 = len(pos), len(neg)

    def psi(xi, xj):
        if xi > xj: return 1.0
        if xi == xj: return 0.5
        return 0.0

    V10 = np.array([np.mean([psi(xi, xj) for xj in neg]) for xi in pos])
    V01 = np.array([np.mean([psi(xi, xj) for xi in pos]) for xj in neg])
    auc = V10.mean()
    return auc, V10, V01, n1, n0


def delong_test(y_true, yp1, yp2):
    """
    DeLong paired AUC test for two models with shared test samples.
    Returns: auc1, auc2, z, p_two_tailed
    """
    auc1, V10_1, V01_1, n1, n0 = structural_components(y_true, yp1)
    auc2, V10_2, V01_2, _, _   = structural_components(y_true, yp2)

    # Covariance matrix of (auc1, auc2)
    S10 = np.cov(np.vstack([V10_1, V10_2]))  # 2×2
    S01 = np.cov(np.vstack([V01_1, V01_2]))  # 2×2

    # Variance of (auc1 - auc2)
    # Var(auc1 - auc2) = Var(auc1) + Var(auc2) - 2*Cov(auc1, auc2)
    # = (1/n1)*(S10[0,0]-2*S10[0,1]+S10[1,1]) + (1/n0)*(S01[0,0]-2*S01[0,1]+S01[1,1])
    # No wait — the correct formula from DeLong 1988:
    # Var(AUC_r - AUC_s) = (1/(n1*n0)) * (n0*(S10_rr - 2*S10_rs + S10_ss)
    #                                      + n1*(S01_rr - 2*S01_rs + S01_ss))
    # where S10_rr = (1/(n1-1)) * sum (V10_r - auc_r)^2, etc.

    # Corrected formula
    s10_cov = np.array([[np.var(V10_1, ddof=1), np.cov(V10_1, V10_2)[0,1]],
                         [np.cov(V10_1, V10_2)[0,1], np.var(V10_2, ddof=1)]])
    s01_cov = np.array([[np.var(V01_1, ddof=1), np.cov(V01_1, V01_2)[0,1]],
                         [np.cov(V01_1, V01_2)[0,1], np.var(V01_2, ddof=1)]])

    diff_var = (s10_cov[0,0] - 2*s10_cov[0,1] + s10_cov[1,1]) / n1 + \
               (s01_cov[0,0] - 2*s01_cov[0,1] + s01_cov[1,1]) / n0

    z = (auc1 - auc2) / np.sqrt(diff_var + 1e-10)
    p = 2 * (1 - norm.cdf(abs(z)))
    return auc1, auc2, z, p


# ═══════════════════════════════════════════════════════════════════════════════
# A. Run LOSO for each model
# ═══════════════════════════════════════════════════════════════════════════════
print("\nRunning LOSO for all LDA models...", flush=True)

t0 = time.time()
yt_ms,   yp_ms   = loso_lda(X_ms,   y); print(f"  MS/LDA: AUC={roc_auc_score(yt_ms, yp_ms):.3f}  [{time.time()-t0:.1f}s]", flush=True)

t0 = time.time()
yt_sp,   yp_sp   = loso_lda(X_sp,   y); print(f"  SP/LDA: AUC={roc_auc_score(yt_sp, yp_sp):.3f}  [{time.time()-t0:.1f}s]", flush=True)

t0 = time.time()
yt_eo,   yp_eo   = loso_lda(X_eo,   y); print(f"  EO/LDA: AUC={roc_auc_score(yt_eo, yp_eo):.3f}  [{time.time()-t0:.1f}s]", flush=True)

t0 = time.time()
yt_mssp, yp_mssp = loso_lda(X_mssp, y); print(f"  MS+SP/LDA: AUC={roc_auc_score(yt_mssp, yp_mssp):.3f}  [{time.time()-t0:.1f}s]", flush=True)

t0 = time.time()
yt_full, yp_full = loso_lda(X_full, y); print(f"  FULL/LDA: AUC={roc_auc_score(yt_full, yp_full):.3f}  [{time.time()-t0:.1f}s]", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B. DeLong paired AUC tests (all share same yt — LOSO labels = y)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("DeLong Paired AUC Tests (4 planned comparisons)")
print("="*60, flush=True)

comparisons = [
    ("SP vs MS",       yp_sp,   yp_ms),
    ("MS+SP vs MS",   yp_mssp,  yp_ms),
    ("MS+SP vs SP",   yp_mssp,  yp_sp),
    ("FULL vs MS+SP", yp_full,  yp_mssp),
]

ALPHA_BONF = 0.0125   # 4 comparisons, α=0.05

results_delong = []
print(f"\n{'Comparison':<20} {'AUC_a':>8} {'AUC_b':>8} {'z':>8} {'p':>10} {'Sig (a=0.0125)':>16}")
print("-"*75)
for cmp_name, ypa, ypb in comparisons:
    auc_a, auc_b, z_, p_ = delong_test(y, ypa, ypb)
    sig = "***" if p_ < 0.001 else ("**" if p_ < 0.01 else ("*" if p_ < ALPHA_BONF else "n.s."))
    print(f"  {cmp_name:<20} {auc_a:>8.3f} {auc_b:>8.3f} {z_:>8.2f} {p_:>10.4f} {sig:>16}")
    results_delong.append({
        "comparison": cmp_name,
        "auc_a": float(auc_a),
        "auc_b": float(auc_b),
        "z": float(z_),
        "p": float(p_),
        "sig": sig,
    })

print(f"\nBonferroni a = {ALPHA_BONF}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# C. Also compute best nested vs SP/LR (for inline text)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Additional: LR models for reference")
print("="*60, flush=True)

from sklearn.linear_model import LogisticRegression

def loso_lr(X, y_bin, C=1.0):
    yt, yp = [], []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[[i]])
        lr = LogisticRegression(solver="lbfgs", C=C, max_iter=2000)
        lr.fit(Xtr, y_bin[tr])
        yt.append(int(y_bin[i]))
        yp.append(float(lr.predict_proba(Xte)[0, 1]))
    return np.array(yt), np.array(yp)

t0 = time.time()
yt_sp_lr, yp_sp_lr = loso_lr(X_sp, y, C=1.0)
auc_sp_lr = roc_auc_score(yt_sp_lr, yp_sp_lr)
print(f"  SP/LR (C=1.0): AUC={auc_sp_lr:.3f}  [{time.time()-t0:.1f}s]", flush=True)

# DeLong: SP/LR vs SP/LDA
auc_a, auc_b, z_, p_ = delong_test(y, yp_sp_lr, yp_sp)
print(f"  DeLong SP/LR vs SP/LDA: z={z_:.2f}  p={p_:.4f}", flush=True)

# Bootstrap CI for SP/LR (paired resampling — single index per iteration)
rng = np.random.default_rng(42)
boot_sp = []
for _ in range(5000):
    idx = rng.integers(0, n, n)          # one index array, used for BOTH yt and yp
    if len(np.unique(yt_sp_lr[idx])) < 2:
        continue
    boot_sp.append(roc_auc_score(yt_sp_lr[idx], yp_sp_lr[idx]))
ci_lo, ci_hi = np.percentile(boot_sp, [2.5, 97.5])
print(f"  SP/LR 95%CI: [{ci_lo:.3f}, {ci_hi:.3f}]", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# D. Save results
# ═══════════════════════════════════════════════════════════════════════════════
out = {
    "lda_aucs": {
        "MS": float(roc_auc_score(yt_ms, yp_ms)),
        "SP": float(roc_auc_score(yt_sp, yp_sp)),
        "EO": float(roc_auc_score(yt_eo, yp_eo)),
        "MS+SP": float(roc_auc_score(yt_mssp, yp_mssp)),
        "FULL": float(roc_auc_score(yt_full, yp_full)),
    },
    "lr_aucs": {
        "SP": float(auc_sp_lr),
        "SP_CI": [float(ci_lo), float(ci_hi)],
    },
    "delong_tests": results_delong,
    "yt_lda_vectors": {
        "MS": yt_ms.tolist(), "SP": yt_sp.tolist(),
        "EO": yt_eo.tolist(), "MS+SP": yt_mssp.tolist(), "FULL": yt_full.tolist(),
    },
    "yp_lda_vectors": {
        "MS": yp_ms.tolist(), "SP": yp_sp.tolist(),
        "EO": yp_eo.tolist(), "MS+SP": yp_mssp.tolist(), "FULL": yp_full.tolist(),
    },
}
with open(RESULTS / "screening_results_delong.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved -> screening_results_delong.json", flush=True)

print("\n" + "="*60)
print("SUMMARY - DeLong Tests")
print("="*60)
for r in results_delong:
    print(f"  {r['comparison']:<20}  AUC_a={r['auc_a']:.3f}  AUC_b={r['auc_b']:.3f}"
          f"  z={r['z']:.2f}  p={r['p']:.4f}  {r['sig']}")
