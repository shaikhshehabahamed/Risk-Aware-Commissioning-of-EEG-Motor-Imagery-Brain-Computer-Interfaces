"""
Script 04f: Compute extra spectral features not in the original feature set.

New features (EC/R02 only):
  [A] 1/f aperiodic exponent at C3, Cz, C4          (3)
  [B] IAF at C3 and C4 (in addition to Cz)           (2)
  [C] IAF lateralisation  (C4_IAF - C3_IAF)          (1)
  [D] Alpha peak amplitude at Cz (above 1/f trend)   (1)
  [E] Within-hemisphere coherence at alpha:
        C3-CP3 (left), C4-CP4 (right)                (2)
  [F] Frontal alpha: F3, Fz, F4 alpha power          (3)
        + F3-F4 lateralisation (frontal asymmetry)   (1)
  [G] Occipital alpha: O1, Oz, O2 alpha power        (3)
        + O1-O2 lateralisation                       (1)
  [H] Parietal alpha: P3, Pz, P4 alpha power         (3)
        + P3-P4 lateralisation                       (1)
                                             Total = 21

Augmented feature matrix:
  SP_aug = SP (47) + extra (21)  = 68 features
  FULL_aug = SP_aug + MS (26) + EO (42) = 136 features

Tests classification with LOSO LR, RF, and their combination.

Reads:  results/features.npz, results/viability_labels.json
        (re-loads raw EEG from DATA_RAW)
Writes: results/features_extra.npz
        results/screening_results_extra.json
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
from scipy.signal import welch, coherence
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESULTS, SFREQ
from src.resting_preprocessing import load_r02

# ── Load existing features ─────────────────────────────────────────────────────
data     = np.load(RESULTS / "features.npz", allow_pickle=True)
subjects = list(data["subjects"])
X_ms     = data["X_ms"].astype(float)
X_sp     = data["X_sp"].astype(float)
X_eo     = data["X_eo"].astype(float)

with open(RESULTS / "viability_labels.json") as f:
    ld = json.load(f)
viable_map = {r["subject"]: int(r["viable"])    for r in ld["per_subject"]}
fold3_map  = {r["subject"]: r["fold3_accuracy"] for r in ld["per_subject"]}

y         = np.array([viable_map[s]  for s in subjects])
fold3_acc = np.array([fold3_map[s]   for s in subjects])
n         = len(y)

# ── PSD helper ─────────────────────────────────────────────────────────────────
NPERSEG  = int(4 * SFREQ)
NOVERLAP = int(NPERSEG * 0.5)

def _psd(data):
    freqs, psd = welch(data, fs=SFREQ, window="hann",
                       nperseg=NPERSEG, noverlap=NOVERLAP, axis=1)
    return freqs, psd

def _bp(psd_row, freqs, lo, hi):
    mask = (freqs >= lo) & (freqs <= hi)
    return float(np.trapz(psd_row[mask], freqs[mask]))

def _lat(psd, freqs, idx_l, idx_r, lo, hi):
    pl = _bp(psd[idx_l], freqs, lo, hi)
    pr = _bp(psd[idx_r], freqs, lo, hi)
    ll, lr = np.log(pl + 1e-30), np.log(pr + 1e-30)
    denom  = abs(ll) + abs(lr)
    return (lr - ll) / denom if denom > 1e-10 else 0.0

def _coh_mean(data, i, j, lo, hi):
    f, c = coherence(data[i], data[j], fs=SFREQ, window="hann",
                     nperseg=NPERSEG, noverlap=NOVERLAP)
    return float(np.mean(c[(f >= lo) & (f <= hi)]))

def _iaf(psd_row, freqs, lo=7.0, hi=14.0):
    """Parabolic interpolation of alpha peak. Returns peak freq or NaN."""
    mask = (freqs >= lo) & (freqs <= hi)
    p = psd_row[mask]; f = freqs[mask]
    pk = int(np.argmax(p))
    if 0 < pk < len(p) - 1:
        y0, y1, y2 = p[pk-1], p[pk], p[pk+1]
        d = y0 - 2*y1 + y2
        delta = float(0.5*(y0-y2)/d) if abs(d) > 1e-30 else 0.0
        df = f[1] - f[0]
        return float(np.clip(f[pk] + delta*df, lo, hi))
    return float(f[pk])

def _aperiodic_exp(psd_row, freqs, excl_lo=7, excl_hi=14, fmin=2, fmax=40):
    """Log-log slope of PSD (1/f exponent), excluding alpha peak."""
    mask = ((freqs >= fmin) & (freqs <= fmax) &
            ~((freqs >= excl_lo) & (freqs <= excl_hi)))
    log_f = np.log(freqs[mask])
    log_p = np.log(np.maximum(psd_row[mask], 1e-30))
    b, _ = np.polyfit(log_f, log_p, 1)
    return float(b)  # negative (typically -2 to -4)


# ── Channel names and indices ──────────────────────────────────────────────────
import mne
mne.set_log_level("WARNING")
from src.resting_preprocessing import get_eeg_channel_names
ch_names = get_eeg_channel_names(subjects[0])
ch_idx   = {ch: i for i, ch in enumerate(ch_names)}
print("Channels available:", ch_names[:10], "...")

required = ["C3", "Cz", "C4", "Cp3", "Cp4", "F3", "Fz", "F4", "O1", "Oz", "O2", "P3", "Pz", "P4"]
missing  = [c for c in required if c not in ch_idx]
if missing:
    print(f"WARNING: Missing channels: {missing}")
    print("Available:", list(ch_idx.keys()))

ix_c3   = ch_idx.get("C3",   ch_idx.get("c3",  0))
ix_cz   = ch_idx.get("Cz",   ch_idx.get("cz",  0))
ix_c4   = ch_idx.get("C4",   ch_idx.get("c4",  0))
ix_cp3  = ch_idx.get("Cp3",  ch_idx.get("cp3", 0))
ix_cp4  = ch_idx.get("Cp4",  ch_idx.get("cp4", 0))
ix_f3   = ch_idx.get("F3",   ch_idx.get("f3",  0))
ix_fz   = ch_idx.get("Fz",   ch_idx.get("fz",  0))
ix_f4   = ch_idx.get("F4",   ch_idx.get("f4",  0))
ix_o1   = ch_idx.get("O1",   ch_idx.get("o1",  0))
ix_oz   = ch_idx.get("Oz",   ch_idx.get("oz",  0))
ix_o2   = ch_idx.get("O2",   ch_idx.get("o2",  0))
ix_p3   = ch_idx.get("P3",   ch_idx.get("p3",  0))
ix_pz   = ch_idx.get("Pz",   ch_idx.get("pz",  0))
ix_p4   = ch_idx.get("P4",   ch_idx.get("p4",  0))


# ── Extract extra features for all subjects ────────────────────────────────────
EXTRA_FEATURES = 21
X_extra = np.full((n, EXTRA_FEATURES), np.nan)

print(f"\nExtracting extra features for {n} subjects...")
t0 = time.time()

for i, subj in enumerate(subjects):
    try:
        ec = load_r02(subj)        # (64, 9600)
        freqs, psd = _psd(ec)

        # [A] Aperiodic exponent at C3, Cz, C4
        ap_c3 = _aperiodic_exp(psd[ix_c3], freqs)
        ap_cz = _aperiodic_exp(psd[ix_cz], freqs)
        ap_c4 = _aperiodic_exp(psd[ix_c4], freqs)

        # [B] IAF at C3, C4
        iaf_c3 = _iaf(psd[ix_c3], freqs)
        iaf_c4 = _iaf(psd[ix_c4], freqs)

        # [C] IAF lateralisation (C4 - C3)
        iaf_lat = iaf_c4 - iaf_c3

        # [D] Alpha peak amplitude at Cz above 1/f trend
        iaf_cz = _iaf(psd[ix_cz], freqs)
        a_mask  = (freqs >= 7) & (freqs <= 14)
        f_iaf_bin = np.argmin(np.abs(freqs - iaf_cz))
        # Expected PSD from 1/f: exp(a + b*log(f))
        b_cz, a_cz = np.polyfit(
            np.log(freqs[((freqs >= 2) & (freqs <= 40) & ~a_mask)]),
            np.log(np.maximum(psd[ix_cz][((freqs >= 2) & (freqs <= 40) & ~a_mask)], 1e-30)), 1)
        expected_at_iaf = float(np.exp(a_cz + b_cz * np.log(freqs[f_iaf_bin])))
        alpha_peak_amp = float(np.log((psd[ix_cz][f_iaf_bin] + 1e-30) / (expected_at_iaf + 1e-30)))

        # [E] Within-hemisphere coherence at alpha
        coh_l = _coh_mean(ec, ix_c3, ix_cp3, 8, 12)
        coh_r = _coh_mean(ec, ix_c4, ix_cp4, 8, 12)

        # [F] Frontal alpha
        fa_f3  = _bp(psd[ix_f3],  freqs, 8, 12)
        fa_fz  = _bp(psd[ix_fz],  freqs, 8, 12)
        fa_f4  = _bp(psd[ix_f4],  freqs, 8, 12)
        fa_lat = _lat(psd, freqs, ix_f3, ix_f4, 8, 12)

        # [G] Occipital alpha
        oa_o1  = _bp(psd[ix_o1], freqs, 8, 12)
        oa_oz  = _bp(psd[ix_oz], freqs, 8, 12)
        oa_o2  = _bp(psd[ix_o2], freqs, 8, 12)
        oa_lat = _lat(psd, freqs, ix_o1, ix_o2, 8, 12)

        # [H] Parietal alpha
        pa_p3  = _bp(psd[ix_p3], freqs, 8, 12)
        pa_pz  = _bp(psd[ix_pz], freqs, 8, 12)
        pa_p4  = _bp(psd[ix_p4], freqs, 8, 12)
        pa_lat = _lat(psd, freqs, ix_p3, ix_p4, 8, 12)

        X_extra[i] = [
            ap_c3, ap_cz, ap_c4,           # 3 aperiodic
            iaf_c3, iaf_c4, iaf_lat,       # 3 IAF
            alpha_peak_amp,                 # 1 peak amplitude
            coh_l, coh_r,                  # 2 within-hemisphere coh
            fa_f3, fa_fz, fa_f4, fa_lat,  # 4 frontal
            oa_o1, oa_oz, oa_o2, oa_lat,  # 4 occipital
            pa_p3, pa_pz, pa_p4, pa_lat,  # 4 parietal
        ]

        if (i + 1) % 20 == 0 or i == n - 1:
            print(f"  [{i+1}/{n}]  {(time.time()-t0):.1f}s", flush=True)

    except Exception as e:
        print(f"  ERROR {subj}: {e}")

print(f"Done in {(time.time()-t0)/60:.1f} min")
print(f"NaN count: {np.isnan(X_extra).sum()}")

# Impute any NaN with column median
for j in range(EXTRA_FEATURES):
    col = X_extra[:, j]
    nans = np.isnan(col)
    if nans.any():
        col[nans] = np.nanmedian(col)

FEATURE_NAMES_EXTRA = [
    "aperiodic_exp_c3", "aperiodic_exp_cz", "aperiodic_exp_c4",
    "iaf_c3", "iaf_c4", "iaf_lat_c4_c3",
    "alpha_peak_amp_cz",
    "coh_alpha_c3_cp3", "coh_alpha_c4_cp4",
    "bp_ec_F3_alpha", "bp_ec_Fz_alpha", "bp_ec_F4_alpha", "lat_frontal_alpha",
    "bp_ec_O1_alpha", "bp_ec_Oz_alpha", "bp_ec_O2_alpha", "lat_occipital_alpha",
    "bp_ec_P3_alpha", "bp_ec_Pz_alpha", "bp_ec_P4_alpha", "lat_parietal_alpha",
]

# Save
np.savez(RESULTS / "features_extra.npz",
         X_extra=X_extra,
         feature_names=np.array(FEATURE_NAMES_EXTRA),
         subjects=np.array(subjects))
print(f"Saved features_extra.npz  shape={X_extra.shape}")


# ── Spearman correlations ──────────────────────────────────────────────────────
from scipy.stats import spearmanr
print("\nSpearman correlations with fold-3 accuracy:")
print(f"  {'Feature':30s}  {'rho':>6s}  {'p':>7s}")
for j, name in enumerate(FEATURE_NAMES_EXTRA):
    rho, pv = spearmanr(X_extra[:, j], fold3_acc)
    marker = " <--" if abs(rho) > 0.20 else ("  *" if pv < 0.05 else "")
    print(f"  {name:30s}  {rho:+6.3f}  {pv:.4f}{marker}")


# ── Classification ─────────────────────────────────────────────────────────────
X_sp_aug   = np.hstack([X_sp, X_extra])           # 47+21=68
X_full_aug = np.hstack([X_ms, X_sp_aug, X_eo])    # 26+68+42=136

def loso_probs(X, y, clf_builder):
    yt, yp = [], []
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if j != i]
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[[i]])
        clf = clf_builder()
        clf.fit(Xtr, y[tr])
        yt.append(int(y[i]))
        yp.append(float(clf.predict_proba(Xte)[0, 1]))
    yt, yp = np.array(yt), np.array(yp)
    return yt, yp, float(roc_auc_score(yt, yp)) if len(np.unique(yt)) == 2 else float("nan")

auc_results = {}
print("\nLOSO classification (augmented features):")
for fs_label, X in [("Extra-21", X_extra), ("SP-aug-68", X_sp_aug), ("FULL-aug-136", X_full_aug)]:
    for clf_name, clf_builder in [
        ("LR",  lambda: LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)),
        ("RF",  lambda: RandomForestClassifier(n_estimators=500, max_features="sqrt",
                                                class_weight="balanced_subsample",
                                                random_state=42, n_jobs=1)),
    ]:
        t0 = time.time()
        yt, yp, a = loso_probs(X, y, clf_builder)
        tag = f"{fs_label}/{clf_name}"
        auc_results[tag] = {"auc": a}
        print(f"  {tag:25s}: AUC={a:.3f}  [{time.time()-t0:.1f}s]")

# Clear subgroup
borderline = (fold3_acc >= 0.50) & (fold3_acc < 0.70)
clear_mask  = ~borderline
y_c         = y[clear_mask]
print(f"\nClear subgroup (n={len(y_c)}) classification:")
for fs_label, X in [("SP-aug-68", X_sp_aug), ("FULL-aug-136", X_full_aug)]:
    X_c = X[clear_mask]
    for clf_name, clf_builder in [
        ("LR", lambda: LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)),
        ("RF", lambda: RandomForestClassifier(n_estimators=500, max_features="sqrt",
                                               class_weight="balanced_subsample",
                                               random_state=42, n_jobs=1)),
    ]:
        t0 = time.time()
        yt, yp = [], []
        for i in range(len(y_c)):
            tr = [j for j in range(len(y_c)) if j != i]
            sc = StandardScaler()
            Xtr = sc.fit_transform(X_c[tr])
            Xte = sc.transform(X_c[[i]])
            clf = clf_builder()
            clf.fit(Xtr, y_c[tr])
            yt.append(int(y_c[i]))
            yp.append(float(clf.predict_proba(Xte)[0, 1]))
        yt, yp = np.array(yt), np.array(yp)
        a = float(roc_auc_score(yt, yp)) if len(np.unique(yt)) == 2 else float("nan")
        tag = f"Clear-{fs_label}/{clf_name}"
        auc_results[tag] = {"auc": a}
        print(f"  {tag:30s}: AUC={a:.3f}  [{time.time()-t0:.1f}s]")

# Summary
print("\n" + "="*60)
print("EXTRA FEATURES SUMMARY")
print("="*60)
for tag, v in sorted(auc_results.items(), key=lambda x: -x[1]["auc"]):
    print(f"  {tag:35s}: AUC={v['auc']:.3f}")

# Save results
out = {"auc_results": auc_results,
       "feature_names": FEATURE_NAMES_EXTRA}
def jsonify(obj):
    if isinstance(obj, dict): return {k: jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list): return [jsonify(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

with open(RESULTS / "screening_results_extra.json", "w") as f:
    json.dump(jsonify(out), f, indent=2)
print(f"\nSaved -> {RESULTS / 'screening_results_extra.json'}")
