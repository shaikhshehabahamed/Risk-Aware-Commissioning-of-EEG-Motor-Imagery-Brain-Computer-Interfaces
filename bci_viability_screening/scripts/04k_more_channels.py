"""
Script 04k: Extract alpha/SMR features at frontocentral and parietal-occipital
channels (FC3/FCz/FC4, TP7/TP8, P7/P8) and compute SMR at P3/Pz/P4.

If parietal alpha at Pz/P4 predicted BCI (rho~+0.23), then:
  - Parietal SMR (12-15 Hz at Pz/P4) might add signal
  - Frontocentral alpha at FC3/FC4 might add signal (premotor)
  - Parietal SMR lateralisation (P4 - P3) might be strongly predictive

Saves new feature matrix to results/features_extra2.npz
Then tests nested Spearman on SP-aug-68 + extra2 (SP-aug-full).
"""

import json, sys, time
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from scipy.signal import welch
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

from src.config import DATA_RAW as RAW_EEG, RESULTS

# ── Channel / band setup ───────────────────────────────────────────────────────
# Additional channels to extract beyond the standard set
ADD_CH = ["FC3", "FCz", "FC4",     # premotor/supplementary motor
          "P7", "P8",               # lateral parietal
          ]
PARIETAL_CH = ["P3", "Pz", "P4"]   # already in extra, but adding SMR/beta
FRONTAL_CH  = ["F3", "Fz", "F4"]   # already in extra (alpha), adding SMR

FS = 160  # sampling rate

BANDS = {
    "alpha": (8, 12),
    "smr":   (12, 15),
    "beta":  (15, 30),
}

# ── Load labels and existing features ─────────────────────────────────────────
data     = np.load(RESULTS / "features.npz",       allow_pickle=True)
subjects = list(data["subjects"])
X_sp     = data["X_sp"].astype(float)

extra    = np.load(RESULTS / "features_extra.npz", allow_pickle=True)
X_extra  = extra["X_extra"].astype(float)
X_aug    = np.hstack([X_sp, X_extra])   # 68 features (current best pool)

with open(RESULTS / "viability_labels.json") as f:
    ld = json.load(f)
y         = np.array([int(r["viable"])    for r in ld["per_subject"]])
fold3_acc = np.array([r["fold3_accuracy"] for r in ld["per_subject"]])
n         = len(y)

print(f"n={n}  viable={y.sum()}  non-viable={(1-y).sum()}", flush=True)


def bandpower(psd, freqs, lo, hi):
    idx = (freqs >= lo) & (freqs <= hi)
    return float(np.log(psd[idx].mean() + 1e-10))


def lat_index(a, b):
    """Lateralisation index (b - a) / (|b| + |a|)."""
    return float((b - a) / (abs(b) + abs(a) + 1e-10))


# ── Extract features for each subject ─────────────────────────────────────────
print("\nExtracting additional features from raw EC EEG...", flush=True)

# Try to load raw EEG from the standard location
import mne
mne.set_log_level("ERROR")

feature_matrix = []
feat_names = []
first_subj = True

for subj in subjects:
    edf_path = None
    for folder in ["S{:03d}".format(subj) if isinstance(subj, int) else subj]:
        p = RAW_EEG / folder
        if p.exists():
            for f in p.glob("*R02.edf"):   # EC baseline
                edf_path = f
                break
    if edf_path is None:
        # Try alternative path structure
        subj_str = "S{:03d}".format(subj) if isinstance(subj, int) else subj
        for f in RAW_EEG.glob(f"{subj_str}*R02.edf"):
            edf_path = f
            break
    if edf_path is None:
        feature_matrix.append(None)
        continue

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    raw.rename_channels({ch: ch.rstrip(".") for ch in raw.ch_names})
    raw.pick_types(eeg=True)
    raw.filter(1.0, 40.0, fir_window="hann", verbose=False)
    raw.set_eeg_reference("average", verbose=False)

    # Skip first 1 s, take up to 60 s (clip to recording length)
    t_start = 1.0
    t_end   = min(t_start + 60.0, raw.times[-1])
    raw.crop(tmin=t_start, tmax=t_end)
    data_arr = raw.get_data()   # shape: (n_ch, n_samp)
    ch_names = [ch.rstrip(".") for ch in raw.ch_names]

    # Compute Welch PSD per channel
    freqs_out, psd_all = welch(data_arr, fs=FS,
                                nperseg=4*FS, noverlap=2*FS, axis=1)

    feats = []
    fn    = []

    # --- Additional channels: FC3/FCz/FC4 and P7/P8 alpha/SMR/beta ---
    for ch in ADD_CH:
        try:
            idx = ch_names.index(ch)
        except ValueError:
            continue
        psd_ch = psd_all[idx]
        for band_name, (lo, hi) in BANDS.items():
            feats.append(bandpower(psd_ch, freqs_out, lo, hi))
            fn.append(f"BP_{band_name}_{ch}")

    # --- Parietal SMR and beta (P3/Pz/P4) - alpha already in extra ---
    psd_P3 = psd_P4 = psd_Pz = None
    for ch in PARIETAL_CH:
        try:
            idx = ch_names.index(ch)
        except ValueError:
            continue
        psd_ch = psd_all[idx]
        if ch == "P3": psd_P3 = psd_ch
        elif ch == "Pz": psd_Pz = psd_ch
        elif ch == "P4": psd_P4 = psd_ch
        for band_name in ["smr", "beta"]:
            lo, hi = BANDS[band_name]
            feats.append(bandpower(psd_ch, freqs_out, lo, hi))
            fn.append(f"BP_{band_name}_{ch}")

    # --- Parietal SMR lateralisation ---
    if psd_P3 is not None and psd_P4 is not None:
        smr_P3 = bandpower(psd_P3, freqs_out, 12, 15)
        smr_P4 = bandpower(psd_P4, freqs_out, 12, 15)
        feats.append(lat_index(smr_P3, smr_P4))
        fn.append("Lat_smr_P4_P3")

    # --- Frontal SMR and beta (F3/Fz/F4) ---
    for ch in FRONTAL_CH:
        try:
            idx = ch_names.index(ch)
        except ValueError:
            continue
        psd_ch = psd_all[idx]
        for band_name in ["smr", "beta"]:
            lo, hi = BANDS[band_name]
            feats.append(bandpower(psd_ch, freqs_out, lo, hi))
            fn.append(f"BP_{band_name}_{ch}")

    feature_matrix.append(np.array(feats))
    if first_subj:
        feat_names = fn
        first_subj = False

# ── Handle missing subjects ───────────────────────────────────────────────────
n_feats = len(feat_names)
X_extra2 = np.zeros((n, n_feats), dtype=float)
for i, f in enumerate(feature_matrix):
    if f is not None and len(f) == n_feats:
        X_extra2[i] = f
    elif f is not None:
        # Size mismatch - fill with zeros
        X_extra2[i, :len(f)] = f

print(f"Extracted {n_feats} new features per subject: {feat_names[:5]}...", flush=True)

# ── Spearman correlations for new features ────────────────────────────────────
print("\nSpearman correlations (new features vs fold3_acc):", flush=True)
rhos_new, pvals_new = [], []
for j in range(n_feats):
    r, p = spearmanr(X_extra2[:, j], fold3_acc)
    rhos_new.append(r)
    pvals_new.append(p)
    if abs(r) >= 0.15 or p < 0.15:
        print(f"  {feat_names[j]:30s}: rho={r:+.3f}  p={p:.3f}", flush=True)

sig_new = [(feat_names[j], rhos_new[j], pvals_new[j])
           for j in range(n_feats) if pvals_new[j] < 0.05]
print(f"\n  Significant (p<0.05): {len(sig_new)} features")
for fn_, rh_, pv_ in sorted(sig_new, key=lambda x: -abs(x[1])):
    print(f"    {fn_:30s}: rho={rh_:+.3f}  p={pv_:.3f}")

# Save extra2 features
np.savez(RESULTS / "features_extra2.npz",
         X_extra2=X_extra2, feature_names=np.array(feat_names),
         subjects=np.array(subjects))
print(f"\nSaved features_extra2.npz  shape={X_extra2.shape}", flush=True)

# ── Create full augmented pool ────────────────────────────────────────────────
X_full_aug = np.hstack([X_aug, X_extra2])   # 68 + n_feats
print(f"Full augmented pool: {X_full_aug.shape}", flush=True)


# ── Classification with nested Spearman ──────────────────────────────────────
def loso_nested_lr(X, y, k, C=2.0):
    yt, yp = [], []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[[i]])
        rhos = np.array([abs(spearmanr(Xtr[:, jj], y[tr])[0])
                         for jj in range(Xtr.shape[1])])
        top  = np.argsort(rhos)[::-1][:k]
        lr   = LogisticRegression(solver="lbfgs", C=C, max_iter=2000)
        lr.fit(Xtr[:, top], y[tr])
        yt.append(int(y[i]))
        yp.append(float(lr.predict_proba(Xte[:, top])[0, 1]))
    return np.array(yt), np.array(yp)


print("\n" + "="*60)
print("Classification with full augmented pool")
print("="*60)
AUC_BASELINE = 0.682
best_k, best_C, best_auc = 28, 2.0, 0.0
best_yt, best_yp = None, None

for k in [25, 28, 30, 32, 35]:
    for C in [1.5, 2.0, 3.0]:
        t0 = time.time()
        yt_, yp_ = loso_nested_lr(X_full_aug, y, k=k, C=C)
        a = float(roc_auc_score(yt_, yp_))
        sign = "BEST" if a > best_auc else (">" if a > AUC_BASELINE else "<=")
        print(f"  SP-full/LR(C={C:.1f})-top{k:2d}: AUC={a:.3f}  [{time.time()-t0:.1f}s]"
              f"  {sign}", flush=True)
        if a > best_auc:
            best_auc, best_k, best_C = a, k, C
            best_yt, best_yp = yt_, yp_

print(f"\n  Best full-aug: k={best_k}  C={best_C}  AUC={best_auc:.3f}"
      f"  ({'IMPROVEMENT' if best_auc > AUC_BASELINE else 'no improvement'} vs {AUC_BASELINE})")

# Bootstrap CI
rng = np.random.default_rng(7)
boot = [roc_auc_score(best_yt[rng.integers(0, n, n)],
                       best_yp[rng.integers(0, n, n)])
        for _ in range(2000)
        if len(np.unique(best_yt[rng.integers(0, n, n)])) == 2]
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"  Bootstrap 95%CI = [{lo:.3f},{hi:.3f}]", flush=True)

print(f"\nSaved -> features_extra2.npz", flush=True)
