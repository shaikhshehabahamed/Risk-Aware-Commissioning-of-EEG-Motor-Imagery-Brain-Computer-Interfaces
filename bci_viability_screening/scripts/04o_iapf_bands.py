"""
Script 04o: Individualized alpha-band (IAB) features.

Instead of fixed 8-12 Hz alpha, compute band power in the
subject-specific alpha window [iAPF-2, iAPF+2] at all sensorimotor and
parietal channels. Also tests theta relative to iAPF.

Motivation: the individual alpha peak frequency (iAPF) already predicts
BCI performance (rho=+0.227, p=0.029). Using subject-specific alpha bands
might boost the alpha-power features by aligning to each subject's actual
oscillatory frequency. Literature: Klimesch (1999) recommends IAB over
fixed bands for alpha-related cognition.

Tests:
  A. IAB features alone (LOSO LR)
  B. SP-aug-68 + IAB features (test if IAB adds to current best)
"""

import json, sys, time
import numpy as np
from pathlib import Path
from scipy.stats import rankdata, spearmanr
from scipy.signal import welch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import mne
mne.set_log_level("ERROR")

try:
    _HERE = Path(__file__).resolve().parent
except NameError:
    _HERE = Path(r"C:\Users\sheha\Downloads\resting_bci_screening\scripts")

PROJ    = _HERE.parent
RESULTS = PROJ / "results"
sys.path.insert(0, str(PROJ))
from src.config import DATA_RAW as RAW_EEG, RESULTS

FS = 160

# ── Load existing features / labels ──────────────────────────────────────────
data     = np.load(RESULTS / "features.npz", allow_pickle=True)
subjects = list(data["subjects"])
X_sp     = data["X_sp"].astype(float)

extra    = np.load(RESULTS / "features_extra.npz", allow_pickle=True)
X_extra  = extra["X_extra"].astype(float)
X_aug    = np.hstack([X_sp, X_extra])

with open(RESULTS / "viability_labels.json") as f:
    ld = json.load(f)
y         = np.array([int(r["viable"])    for r in ld["per_subject"]])
fold3_acc = np.array([r["fold3_accuracy"] for r in ld["per_subject"]])
n         = len(y)
print(f"n={n}  viable={y.sum()}", flush=True)

# ── Channel lists ─────────────────────────────────────────────────────────────
CHANNELS = ["C3", "Cz", "C4", "CP3", "CPz", "CP4", "P3", "Pz", "P4",
            "F3", "Fz", "F4", "O1", "Oz", "O2"]


def bandpower(psd, freqs, lo, hi):
    idx = (freqs >= lo) & (freqs <= hi)
    if idx.sum() == 0:
        return 0.0
    return float(np.log(psd[idx].mean() + 1e-10))


def iapf_at_channel(psd, freqs, lo=7, hi=14):
    """Parabolic interpolation of the peak in [lo, hi] Hz."""
    idx = np.where((freqs >= lo) & (freqs <= hi))[0]
    if len(idx) < 3:
        return 10.0
    peak_idx = idx[np.argmax(psd[idx])]
    if peak_idx == 0 or peak_idx == len(psd) - 1:
        return float(freqs[peak_idx])
    y0, y1, y2 = psd[peak_idx - 1], psd[peak_idx], psd[peak_idx + 1]
    denom = 2 * (2 * y1 - y0 - y2)
    if denom == 0:
        return float(freqs[peak_idx])
    offset = (y2 - y0) / denom
    return float(freqs[peak_idx] + offset * (freqs[1] - freqs[0]))


# ── Extract IAB features per subject ─────────────────────────────────────────
print("\nExtracting IAB features from raw EC EEG...", flush=True)
feature_matrix = []
feat_names     = []
first_subj     = True

for subj in subjects:
    subj_str = "S{:03d}".format(subj) if isinstance(subj, int) else str(subj)
    edf_path  = None
    for f in RAW_EEG.glob(f"{subj_str}R02.edf"):
        edf_path = f; break
    if edf_path is None:
        for f in (RAW_EEG / subj_str).glob("*R02.edf"):
            edf_path = f; break
    if edf_path is None:
        feature_matrix.append(None); continue

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    raw.rename_channels({ch: ch.rstrip(".") for ch in raw.ch_names})
    raw.pick_types(eeg=True)
    raw.filter(1.0, 40.0, fir_window="hann", verbose=False)
    raw.set_eeg_reference("average", verbose=False)
    t_end = min(61.0, raw.times[-1])
    raw.crop(tmin=1.0, tmax=t_end)
    data_arr = raw.get_data()
    ch_names = [ch.rstrip(".") for ch in raw.ch_names]

    freqs_out, psd_all = welch(data_arr, fs=FS, nperseg=4*FS, noverlap=2*FS, axis=1)

    # Get iAPF from Cz
    try:
        cz_idx = ch_names.index("Cz")
        iapf = iapf_at_channel(psd_all[cz_idx], freqs_out)
    except ValueError:
        iapf = 10.0
    iapf = max(7.0, min(13.0, iapf))   # clamp to sane range
    iab_lo, iab_hi = iapf - 2.0, iapf + 2.0

    feats = []
    fn    = []

    for ch in CHANNELS:
        try:
            idx = ch_names.index(ch)
        except ValueError:
            feats.append(0.0); fn.append(f"IAB_{ch}")
            continue
        psd_ch = psd_all[idx]
        bp = bandpower(psd_ch, freqs_out, iab_lo, iab_hi)
        feats.append(bp); fn.append(f"IAB_{ch}")

    # IAB lateralisation
    try:
        c3_bp = bandpower(psd_all[ch_names.index("C3")], freqs_out, iab_lo, iab_hi)
        c4_bp = bandpower(psd_all[ch_names.index("C4")], freqs_out, iab_lo, iab_hi)
        feats.append((c4_bp - c3_bp) / (abs(c4_bp) + abs(c3_bp) + 1e-10))
        fn.append("IAB_lat_C4_C3")
        p3_bp = bandpower(psd_all[ch_names.index("P3")], freqs_out, iab_lo, iab_hi)
        p4_bp = bandpower(psd_all[ch_names.index("P4")], freqs_out, iab_lo, iab_hi)
        feats.append((p4_bp - p3_bp) / (abs(p4_bp) + abs(p3_bp) + 1e-10))
        fn.append("IAB_lat_P4_P3")
    except ValueError:
        feats += [0.0, 0.0]; fn += ["IAB_lat_C4_C3", "IAB_lat_P4_P3"]

    # iAPF itself (redundant with extra but include for completeness)
    feats.append(iapf); fn.append("iAPF_Cz_recomputed")

    feature_matrix.append(np.array(feats))
    if first_subj:
        feat_names = fn
        first_subj = False

n_feats = len(feat_names)
X_iab = np.zeros((n, n_feats), dtype=float)
for i, f in enumerate(feature_matrix):
    if f is not None and len(f) == n_feats:
        X_iab[i] = f

print(f"IAB features: {n_feats}  ({feat_names[:5]}...)", flush=True)

# ── Spearman correlations ─────────────────────────────────────────────────────
print("\nSpearman correlations (IAB features vs fold3_acc):", flush=True)
for j in range(n_feats):
    r, p = spearmanr(X_iab[:, j], fold3_acc)
    if abs(r) >= 0.10 or p < 0.20:
        print(f"  {feat_names[j]:25s}: rho={r:+.3f}  p={p:.3f}", flush=True)


# ── Nested Spearman helper ───────────────────────────────────────────────────
def spearman_top_k(X_tr, y_tr, k):
    Xr = np.apply_along_axis(rankdata, 0, X_tr)
    yr = rankdata(y_tr)
    Xr -= Xr.mean(0); yr -= yr.mean()
    rho = np.abs(Xr.T @ yr) / (np.linalg.norm(Xr, axis=0) * np.linalg.norm(yr) + 1e-10)
    return np.argsort(rho)[::-1][:k]

def loso_nested(X, y_bin, k, C=2.0):
    yt, yp = [], []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[[i]])
        top = spearman_top_k(Xtr, y_bin[tr], k)
        lr  = LogisticRegression(solver="lbfgs", C=C, max_iter=2000)
        lr.fit(Xtr[:, top], y_bin[tr])
        yt.append(int(y_bin[i]))
        yp.append(float(lr.predict_proba(Xte[:, top])[0, 1]))
    return np.array(yt), np.array(yp)


# ═══════════════════════════════════════════════════════════════════════════════
# A. IAB features alone (LOSO)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("A. IAB features alone (nested LOSO)")
print("="*60, flush=True)

best_iab = 0.0
for k in [5, 8, 10, 12, 15]:
    for C in [1.0, 2.0]:
        t0 = time.time()
        yt_, yp_ = loso_nested(X_iab, y, k=k, C=C)
        a = float(roc_auc_score(yt_, yp_))
        print(f"  IAB/LR-top{k:2d} C={C}: AUC={a:.3f}  [{time.time()-t0:.1f}s]",
              flush=True)
        if a > best_iab:
            best_iab = a

print(f"\n  A-best IAB: AUC={best_iab:.3f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# B. SP-aug-68 + IAB (combined pool, nested)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("B. SP-aug-68 + IAB combined pool (nested)")
print("="*60, flush=True)

X_comb = np.hstack([X_aug, X_iab])  # 68 + n_feats
BASELINE = 0.682

print(f"Combined pool: {X_comb.shape[1]} features", flush=True)

best_comb = 0.0
for k in [28, 30, 32, 35]:
    for C in [1.5, 2.0, 2.5]:
        t0 = time.time()
        yt_, yp_ = loso_nested(X_comb, y, k=k, C=C)
        a = float(roc_auc_score(yt_, yp_))
        sig = "IMPROVEMENT!" if a > BASELINE else ("<=baseline", "")[a > BASELINE]
        print(f"  comb/LR-top{k:2d} C={C}: AUC={a:.3f}  [{time.time()-t0:.1f}s]  "
              f"{'IMPROVEMENT!' if a > BASELINE else ''}", flush=True)
        if a > best_comb:
            best_comb = a

print(f"\n  B-best combined: AUC={best_comb:.3f}  "
      f"({'IMPROVEMENT' if best_comb > BASELINE else 'no gain'} vs {BASELINE})",
      flush=True)

print("\nDONE", flush=True)
