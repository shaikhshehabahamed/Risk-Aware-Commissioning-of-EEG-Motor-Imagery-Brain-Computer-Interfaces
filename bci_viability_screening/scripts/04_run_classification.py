"""
Script 04: LOSO classification on all feature sets.

Feature sets evaluated:
  MS       — 26 microstate features (R02)
  SP       — 47 spectral features   (R02 eyes-closed)
  EO       — 42 eyes-open + reactivity features (R01+R02)
  MS+SP    — 73 combined
  MS+SP+EO — 115 full feature set

Classifiers: LDA (shrinkage), SVM (RBF, nested-CV), LR (L2)
Primary metric: AUC-ROC with 10,000-resample bootstrap CI
Statistical tests: DeLong paired AUC, Bonferroni alpha = 0.05/4 = 0.0125
Permutation test: 1,000 shuffles of y -> empirical p-value

Reads:  results/features.npz, results/viability_labels.json
Writes: results/screening_results.json
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESULTS
from src.classification import (
    loso_cv, loso_cv_lda_fast, compute_metrics, bootstrap_auc_ci, delong_test,
    lda_feature_importance, spearman_correlations,
)
from src.microstate_features import FEATURE_NAMES_MICROSTATE
from src.spectral_features import FEATURE_NAMES_SPECTRAL, FEATURE_NAMES_EO

N_BOOTSTRAP     = 10_000
N_PERMUTATIONS  = 1_000
BONFERRONI_SETS = 4     # MS vs SP, MS+SP vs MS, MS+SP vs SP, full vs MS+SP
BONFERRONI_A    = 0.05 / BONFERRONI_SETS   # 0.0125

# ── Load features ──────────────────────────────────────────────────────────────
feat_path = RESULTS / "features.npz"
if not feat_path.exists():
    raise FileNotFoundError(f"Run 03_extract_features.py first.")

data      = np.load(feat_path, allow_pickle=True)
subjects  = list(data["subjects"])
X_ms      = data["X_ms"].astype(float)
X_sp      = data["X_sp"].astype(float)
X_eo      = data["X_eo"].astype(float)

X_mssp    = np.hstack([X_ms, X_sp])
X_full    = np.hstack([X_ms, X_sp, X_eo])

# Always load y fresh from viability_labels.json (may differ from cached y
# if the label script was re-run with a different task).
label_path = RESULTS / "viability_labels.json"
with open(label_path) as f:
    label_data = json.load(f)

viable_map = {r["subject"]: int(r["viable"])         for r in label_data["per_subject"]}
fold3_map  = {r["subject"]: r["fold3_accuracy"]      for r in label_data["per_subject"]}

y         = np.array([viable_map[s] for s in subjects])
fold3_acc = np.array([fold3_map[s]  for s in subjects])

print(f"Task: {label_data.get('task','?')}  Runs: {label_data.get('runs','?')}")
print(f"Subjects: {len(subjects)}")
print(f"Viable: {y.sum()}  Non-viable: {(1-y).sum()}")
print(f"Shapes: MS={X_ms.shape}  SP={X_sp.shape}  EO={X_eo.shape}")
print(f"  MS+SP={X_mssp.shape}  Full={X_full.shape}")

# ── Feature sets ───────────────────────────────────────────────────────────────
feature_sets = {
    "MS":       (X_ms,   "Microstate (26)"),
    "SP":       (X_sp,   "Spectral-EC (47)"),
    "EO":       (X_eo,   "Spectral-EO+Reactivity (42)"),
    "MS+SP":    (X_mssp, "MS + Spectral-EC (73)"),
    "FULL":     (X_full, "MS + SP + EO (115)"),
}

# ── LOSO cross-validation ─────────────────────────────────────────────────────
print("\n" + "="*65)
print("Running LOSO cross-validation...")
t0 = time.time()

loso_results = {}
for fs_label, (X, desc) in feature_sets.items():
    print(f"  {fs_label}: {desc} ...", end=" ", flush=True)
    loso_results[fs_label] = loso_cv(X, y)
    lda_auc = roc_auc_score(
        loso_results[fs_label]["LDA"]["y_true"],
        loso_results[fs_label]["LDA"]["y_prob"],
    )
    print(f"LDA AUC={lda_auc:.3f}")

print(f"\nLOSO done in {(time.time()-t0)/60:.1f} min")

# ── Metrics + bootstrap CIs ───────────────────────────────────────────────────
print("\n" + "="*65)
print("Computing metrics and bootstrap CIs...")

results_table = {}
for fs_label in feature_sets:
    results_table[fs_label] = {}
    for clf_label in ["LDA", "SVM", "LR"]:
        yt = np.array(loso_results[fs_label][clf_label]["y_true"])
        yp = np.array(loso_results[fs_label][clf_label]["y_prob"])
        metrics    = compute_metrics(yt, yp)
        lo, hi     = bootstrap_auc_ci(yt, yp, n_resamples=N_BOOTSTRAP)
        results_table[fs_label][clf_label] = {**metrics, "auc_ci_lo": lo, "auc_ci_hi": hi}
        print(f"  {fs_label:8s}/{clf_label}: AUC={metrics['auc']:.3f} [{lo:.3f},{hi:.3f}]  "
              f"BalAcc={metrics['bal_acc']:.3f}  Sen={metrics['sensitivity']:.3f}  "
              f"Spec={metrics['specificity']:.3f}")

# ── Permutation test (LDA, each feature set) ──────────────────────────────────
print("\n" + "="*65)
print(f"Permutation test (n={N_PERMUTATIONS} shuffles, LDA)...")

perm_results = {}
rng = np.random.default_rng(0)
yt_true = np.array(loso_results["MS"]["LDA"]["y_true"])

for fs_label, (X, _) in feature_sets.items():
    obs_auc = results_table[fs_label]["LDA"]["auc"]
    perm_aucs = []
    t_perm = time.time()
    for perm_i in range(N_PERMUTATIONS):
        y_perm = rng.permutation(y)
        # Use the LDA-only fast path (no SVM/LR overhead) for permutation
        perm_aucs.append(loso_cv_lda_fast(X, y_perm))
        if (perm_i + 1) % 100 == 0:
            elapsed = time.time() - t_perm
            eta = elapsed / (perm_i + 1) * (N_PERMUTATIONS - perm_i - 1)
            print(f"    [{fs_label}] {perm_i+1}/{N_PERMUTATIONS}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)
    perm_aucs = np.array(perm_aucs)
    p_perm = float((perm_aucs >= obs_auc).mean())
    perm_results[fs_label] = {
        "observed_auc": obs_auc,
        "perm_mean":    float(perm_aucs.mean()),
        "perm_std":     float(perm_aucs.std()),
        "p_value":      p_perm,
    }
    sig = "**" if p_perm < 0.05 else "n.s."
    print(f"  {fs_label:8s}: AUC={obs_auc:.3f}  perm_mean={perm_aucs.mean():.3f}  "
          f"p={p_perm:.3f} {sig}")

# ── DeLong paired AUC tests ───────────────────────────────────────────────────
print("\n" + "="*65)
print(f"DeLong paired AUC tests (LDA, Bonferroni alpha={BONFERRONI_A:.4f}):")

yt = np.array(loso_results["MS"]["LDA"]["y_true"])
delong_comparisons = {
    "MS_vs_SP":       ("MS",    "SP"),
    "MSSP_vs_MS":     ("MS+SP", "MS"),
    "MSSP_vs_SP":     ("MS+SP", "SP"),
    "FULL_vs_MSSP":   ("FULL",  "MS+SP"),
}

delong_results = {}
for comp, (a, b) in delong_comparisons.items():
    yp_a = np.array(loso_results[a]["LDA"]["y_prob"])
    yp_b = np.array(loso_results[b]["LDA"]["y_prob"])
    res  = delong_test(yt, yp_a, yp_b)
    sig  = "** SIG **" if res["p_value"] < BONFERRONI_A else "n.s."
    print(f"  {comp:18s}: AUC_A={res['auc_a']:.3f} AUC_B={res['auc_b']:.3f} "
          f"z={res['z']:+.3f} p={res['p_value']:.4f} {sig}")
    delong_results[comp] = res

# ── Feature importance ─────────────────────────────────────────────────────────
print("\n" + "="*65)
print("LDA feature importance and Spearman correlations...")

fn_ms   = list(FEATURE_NAMES_MICROSTATE)
fn_sp   = list(FEATURE_NAMES_SPECTRAL)
fn_eo   = list(FEATURE_NAMES_EO)
fn_mssp = fn_ms + fn_sp
fn_full = fn_ms + fn_sp + fn_eo

imp_ms   = lda_feature_importance(X_ms,   y)
imp_sp   = lda_feature_importance(X_sp,   y)
imp_mssp = lda_feature_importance(X_mssp, y)
imp_full = lda_feature_importance(X_full, y)

spearman_ms = spearman_correlations(X_ms, fold3_acc)
spearman_sp = spearman_correlations(X_sp, fold3_acc)

print("\n  Top-5 MS features (LDA importance):")
top5_ms = np.argsort(imp_ms)[::-1][:5]
for idx in top5_ms:
    print(f"    {fn_ms[idx]:30s}  imp={imp_ms[idx]:.3f}  "
          f"rho={spearman_ms['rho'][idx]:+.3f}  p={spearman_ms['p_values'][idx]:.4f}")

print("\n  Top-5 SP features (LDA importance):")
top5_sp = np.argsort(imp_sp)[::-1][:5]
for idx in top5_sp:
    print(f"    {fn_sp[idx]:35s}  imp={imp_sp[idx]:.3f}  "
          f"rho={spearman_sp['rho'][idx]:+.3f}  p={spearman_sp['p_values'][idx]:.4f}")

print("\n  Significant Spearman correlates (MS, Bonferroni):")
for idx in spearman_ms["significant_idx"]:
    print(f"    {fn_ms[idx]:30s}  rho={spearman_ms['rho'][idx]:+.3f}  "
          f"p={spearman_ms['p_values'][idx]:.4f}")

print("\n  Significant Spearman correlates (SP, Bonferroni):")
for idx in spearman_sp["significant_idx"]:
    print(f"    {fn_sp[idx]:35s}  rho={spearman_sp['rho'][idx]:+.3f}  "
          f"p={spearman_sp['p_values'][idx]:.4f}")

# ── Sensitivity analysis ───────────────────────────────────────────────────────
print("\n" + "="*65)
print("Threshold sensitivity analysis (FULL/LDA)...")

with open(label_path) as f:
    label_raw = json.load(f)

f3_per_subj = {r["subject"]: r["fold3_accuracy"] for r in label_raw["per_subject"]}
f3_arr = np.array([f3_per_subj[s] for s in subjects])

sensitivity_results = {}
for thresh in [0.55, 0.60, 0.65]:
    y_t = (f3_arr >= thresh).astype(int)
    if len(np.unique(y_t)) < 2:
        continue
    lo_t = loso_cv(X_full, y_t)
    yt2  = np.array(lo_t["LDA"]["y_true"])
    yp2  = np.array(lo_t["LDA"]["y_prob"])
    auc2 = float(roc_auc_score(yt2, yp2)) if len(np.unique(yt2)) == 2 else float("nan")
    n_v  = int(y_t.sum())
    print(f"  Threshold={thresh:.0%}: n_viable={n_v}  FULL/LDA AUC={auc2:.3f}")
    sensitivity_results[str(thresh)] = {
        "threshold": thresh, "n_viable": n_v,
        "n_nonviable": int((1-y_t).sum()), "full_lda_auc": auc2,
    }

# ── Save ───────────────────────────────────────────────────────────────────────
output = {
    "n_subjects":    len(subjects),
    "n_viable":      int(y.sum()),
    "n_nonviable":   int((1-y).sum()),
    "bonferroni_alpha": BONFERRONI_A,
    "n_bootstrap":   N_BOOTSTRAP,
    "n_permutations": N_PERMUTATIONS,

    "results_table": results_table,
    "delong":        delong_results,
    "permutation":   perm_results,

    "feature_importance": {
        "MS":    imp_ms.tolist(),
        "SP":    imp_sp.tolist(),
        "MS+SP": imp_mssp.tolist(),
        "FULL":  imp_full.tolist(),
        "feature_names_ms":   fn_ms,
        "feature_names_sp":   fn_sp,
        "feature_names_eo":   fn_eo,
        "feature_names_full": fn_full,
    },

    "spearman": {
        "MS": {k: (v.tolist() if hasattr(v, "tolist") else v)
               for k, v in spearman_ms.items()},
        "SP": {k: (v.tolist() if hasattr(v, "tolist") else v)
               for k, v in spearman_sp.items()},
        "feature_names_ms": fn_ms,
        "feature_names_sp": fn_sp,
    },

    "sensitivity_analysis": sensitivity_results,

    "loso_probs": {
        fs: {clf: {"y_true": v["y_true"], "y_prob": v["y_prob"]}
             for clf, v in clf_dict.items()}
        for fs, clf_dict in loso_results.items()
    },

    "subjects":         subjects,
    "fold3_accuracies": fold3_acc.tolist(),
}

out_path = RESULTS / "screening_results.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved -> {out_path}")
print(f"Total runtime: {(time.time()-t0)/60:.1f} minutes")
