"""
Script 04m: XGBoost and ensemble tests on SP-aug-68.

Tests:
  A. XGBoost with nested Spearman selection (like LR but with XGB)
  B. Ensemble: average calibrated LR + XGB probabilities
  C. RBF-SVM with nested selection
  D. Nested XGB on MS features (known non-linear signal from 04h)
  E. Ridge regression → threshold → AUC (already 0.631, baseline)
  F. Best SP-aug-68/LR-nested benchmark for comparison
"""

import json, sys, time
import numpy as np
from pathlib import Path
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

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

extra    = np.load(RESULTS / "features_extra.npz", allow_pickle=True)
X_extra  = extra["X_extra"].astype(float)
X_aug    = np.hstack([X_sp, X_extra])   # 68 features

with open(RESULTS / "viability_labels.json") as f:
    ld = json.load(f)
y = np.array([int(r["viable"]) for r in ld["per_subject"]])
n = len(y)

print(f"n={n}  viable={y.sum()}  non-viable={(1-y).sum()}", flush=True)

# ── Vectorised Spearman ───────────────────────────────────────────────────────
def spearman_top_k(X_tr, y_tr, k):
    Xr = np.apply_along_axis(rankdata, 0, X_tr)
    yr = rankdata(y_tr)
    Xr -= Xr.mean(0); yr -= yr.mean()
    rho = np.abs(Xr.T @ yr) / (np.linalg.norm(Xr, axis=0) * np.linalg.norm(yr) + 1e-10)
    return np.argsort(rho)[::-1][:k]


def loso_nested(X, y_bin, clf_fn, k, scale=True):
    """LOSO with nested Spearman selection; clf_fn(X_tr, y_tr) -> clf."""
    yt, yp = [], []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        if scale:
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[tr])
            Xte = sc.transform(X[[i]])
        else:
            Xtr, Xte = X[tr], X[[i]]
        if k is not None:
            top  = spearman_top_k(Xtr, y_bin[tr], k)
            Xtr_ = Xtr[:, top]; Xte_ = Xte[:, top]
        else:
            Xtr_, Xte_ = Xtr, Xte
        clf = clf_fn(Xtr_, y_bin[tr])
        yt.append(int(y_bin[i]))
        if hasattr(clf, "predict_proba"):
            yp.append(float(clf.predict_proba(Xte_)[0, 1]))
        else:
            yp.append(float(clf.decision_function(Xte_)[0]))
    return np.array(yt), np.array(yp)


BEST_LR = 0.682

print("\n" + "="*60)
print("BENCHMARK — SP-aug-68/LR-nested (C=2.0, k=28)  [0.682]")
print("="*60, flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# A. XGBoost with nested Spearman selection on SP-aug-68
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("A. Nested XGBoost on SP-aug-68")
print("="*60, flush=True)

def make_xgb(n_est=200, lr=0.05, max_d=3, sub=0.8, col_bt=0.8, gamma=0.1,
             min_child=2, l2=1.0):
    def _fn(Xtr, ytr):
        clf = XGBClassifier(
            n_estimators=n_est, learning_rate=lr, max_depth=max_d,
            subsample=sub, colsample_bytree=col_bt, gamma=gamma,
            min_child_weight=min_child, reg_lambda=l2,
            use_label_encoder=False, eval_metric="logloss",
            verbosity=0, random_state=42,
        )
        clf.fit(Xtr, ytr)
        return clf
    return _fn

best_xgb_auc = 0.0
best_xgb_cfg = {}
for k in [15, 20, 28]:
    for n_est, lr, max_d in [(100, 0.1, 3), (200, 0.05, 3), (300, 0.05, 4),
                              (200, 0.1, 2), (100, 0.05, 4)]:
        t0 = time.time()
        yt_, yp_ = loso_nested(X_aug, y, make_xgb(n_est=n_est, lr=lr, max_d=max_d), k=k)
        a = float(roc_auc_score(yt_, yp_))
        cmp = "BEST" if a > best_xgb_auc else (">LR" if a > BEST_LR else "<=LR")
        print(f"  XGB k={k:2d} n={n_est} lr={lr} d={max_d}: AUC={a:.3f}  "
              f"[{time.time()-t0:.1f}s]  {cmp}", flush=True)
        if a > best_xgb_auc:
            best_xgb_auc = a
            best_xgb_cfg = dict(k=k, n_est=n_est, lr=lr, max_d=max_d)
            best_yt_xgb, best_yp_xgb = yt_.copy(), yp_.copy()

print(f"\n  A-best: {best_xgb_cfg}  AUC={best_xgb_auc:.3f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B. Ensemble: LR + XGBoost average
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("B. Ensemble LR + XGBoost (LOSO probabilities averaged)")
print("="*60, flush=True)

# Re-run LR LOSO to get probabilities
def make_lr(C=2.0):
    def _fn(Xtr, ytr):
        clf = LogisticRegression(solver="lbfgs", C=C, max_iter=2000)
        clf.fit(Xtr, ytr)
        return clf
    return _fn

t0 = time.time()
yt_lr, yp_lr = loso_nested(X_aug, y, make_lr(C=2.0), k=28)
auc_lr = float(roc_auc_score(yt_lr, yp_lr))
print(f"  LR k=28 C=2.0: AUC={auc_lr:.3f}  [{time.time()-t0:.1f}s]", flush=True)

# Ensemble with best XGB
if best_xgb_auc > 0.5:
    # Get XGB preds at best config
    t0 = time.time()
    yt_xb, yp_xb = loso_nested(
        X_aug, y,
        make_xgb(n_est=best_xgb_cfg["n_est"], lr=best_xgb_cfg["lr"],
                 max_d=best_xgb_cfg["max_d"]),
        k=best_xgb_cfg["k"])
    auc_xb = float(roc_auc_score(yt_xb, yp_xb))
    # Rank-normalise before averaging
    from scipy.stats import rankdata as rdata
    yp_lr_rank  = rdata(yp_lr) / n
    yp_xb_rank  = rdata(yp_xb) / n

    for w in [0.5, 0.6, 0.7, 0.8]:
        yp_ens = w * yp_lr_rank + (1 - w) * yp_xb_rank
        a = float(roc_auc_score(yt_lr, yp_ens))
        print(f"  Ensemble w_LR={w:.1f}: AUC={a:.3f}", flush=True)

    # Also try raw prob average
    yp_raw_ens = 0.6 * yp_lr + 0.4 * yp_xb
    a_raw = float(roc_auc_score(yt_lr, yp_raw_ens))
    print(f"  Ensemble raw 60/40: AUC={a_raw:.3f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# C. RBF-SVM with nested Spearman
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("C. RBF-SVM with nested Spearman on SP-aug-68")
print("="*60, flush=True)

def make_svm(C=1.0, gamma="scale"):
    def _fn(Xtr, ytr):
        clf = SVC(C=C, kernel="rbf", gamma=gamma, probability=True, random_state=42)
        clf.fit(Xtr, ytr)
        return clf
    return _fn

best_svm = 0.0
for k in [20, 28]:
    for C in [0.5, 1.0, 2.0, 5.0]:
        t0 = time.time()
        yt_, yp_ = loso_nested(X_aug, y, make_svm(C=C), k=k)
        a = float(roc_auc_score(yt_, yp_))
        cmp = "BEST" if a > best_svm else (">LR" if a > BEST_LR else "")
        print(f"  SVM k={k:2d} C={C}: AUC={a:.3f}  [{time.time()-t0:.1f}s]  {cmp}",
              flush=True)
        if a > best_svm:
            best_svm = a

print(f"\n  C-best SVM: AUC={best_svm:.3f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# D. XGBoost on MS features (non-linear signal)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("D. Nested XGBoost on MS features")
print("="*60, flush=True)

def loso_no_selection(X, y_bin, clf_fn):
    yt, yp = [], []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[[i]])
        clf = clf_fn(Xtr, y_bin[tr])
        yt.append(int(y_bin[i]))
        if hasattr(clf, "predict_proba"):
            yp.append(float(clf.predict_proba(Xte)[0, 1]))
        else:
            yp.append(float(clf.decision_function(Xte)[0]))
    return np.array(yt), np.array(yp)

ms_best = 0.0
for n_est, lr, max_d in [(100, 0.1, 3), (200, 0.05, 4), (200, 0.1, 2)]:
    t0 = time.time()
    yt_, yp_ = loso_no_selection(X_ms, y, make_xgb(n_est=n_est, lr=lr, max_d=max_d))
    a = float(roc_auc_score(yt_, yp_))
    print(f"  MS/XGB n={n_est} lr={lr} d={max_d}: AUC={a:.3f}  [{time.time()-t0:.1f}s]",
          flush=True)
    if a > ms_best:
        ms_best = a
        ms_yt, ms_yp = yt_.copy(), yp_.copy()

print(f"\n  D-best MS/XGB: AUC={ms_best:.3f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# E. 3-way ensemble: LR + XGB + MS/XGB
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("E. 3-way ensemble: SP-aug/LR + SP-aug/XGB + MS/XGB")
print("="*60, flush=True)

from scipy.stats import rankdata as rdata

yp_lr_r  = rdata(yp_lr)  / n
yp_xb_r  = rdata(yp_xb)  / n if best_xgb_auc > 0.5 else yp_lr_r
yp_ms_r  = rdata(ms_yp)  / n

for w_lr, w_sp, w_ms in [(0.6, 0.2, 0.2), (0.5, 0.3, 0.2), (0.7, 0.2, 0.1),
                          (0.5, 0.5, 0.0), (0.8, 0.2, 0.0)]:
    yp_3ens = w_lr * yp_lr_r + w_sp * yp_xb_r + w_ms * yp_ms_r
    a = float(roc_auc_score(yt_lr, yp_3ens))
    print(f"  w_LR={w_lr} w_XGB={w_sp} w_MS={w_ms}: AUC={a:.3f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  Benchmark SP-aug/LR-nested:  AUC={auc_lr:.3f}")
print(f"  A: SP-aug/XGB-nested best:   AUC={best_xgb_auc:.3f}  cfg={best_xgb_cfg}")
print(f"  C: SP-aug/SVM-nested best:   AUC={best_svm:.3f}")
print(f"  D: MS/XGB (no selection):    AUC={ms_best:.3f}")
print(f"  Best single model:           {max(auc_lr, best_xgb_auc, best_svm):.3f}")
