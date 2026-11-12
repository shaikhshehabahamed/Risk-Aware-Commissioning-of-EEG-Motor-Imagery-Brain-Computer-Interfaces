"""
Script 03: Extract all resting-state features.

Features per subject
--------------------
Microstate (MS) — 26 features from R02:
  temporal stats (duration, occurrence, coverage) for K=4 states,
  transition probability matrix (off-diagonal, row-normalised),
  GFP mean + std.

Spectral-EC (SP) — 47 features from R02 (eyes-closed):
  band powers at C3/Cz/C4/CP3/CPz/CP4 x 6 bands,
  C3-C4 and CP3-CP4 lateralisation (alpha/SMR/beta),
  C3-C4 inter-hemispheric coherence (alpha, beta),
  individual alpha peak frequency, spectral entropy, theta/alpha ratio.

Spectral-EO (EO) — 42 features from R01 (eyes-open) + reactivity:
  same band powers / lateralisation for R01,
  log alpha-reactivity (R02/R01) at C3/Cz/C4.

Grand total: 26 + 47 + 42 = 115 features.

Reads:  results/subject_list.json, results/viability_labels.json
Writes: results/microstate_templates.npy  (cached; skipped if exists)
        results/microstate_templates_meta.json
        results/features.npz
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESULTS
from src.resting_preprocessing import load_r02, load_r01, get_eeg_channel_names
from src.microstate_features import (
    fit_microstate_templates, get_template_maps, compute_microstate_features,
    FEATURE_NAMES_MICROSTATE,
)
from src.spectral_features import (
    compute_spectral_features, compute_eo_features,
    FEATURE_NAMES_SPECTRAL, FEATURE_NAMES_EO,
)

# ── Load metadata ──────────────────────────────────────────────────────────────
subj_path = RESULTS / "subject_list.json"
if not subj_path.exists():
    raise FileNotFoundError(f"Missing: {subj_path}")

with open(subj_path) as f:
    subj_data = json.load(f)
subjects = subj_data["included_subjects"]

label_path = RESULTS / "viability_labels.json"
if not label_path.exists():
    raise FileNotFoundError(f"Missing: {label_path}")

with open(label_path) as f:
    label_data = json.load(f)
viability_map = {r["subject"]: int(r["viable"]) for r in label_data["per_subject"]}

print(f"Subjects to process: {len(subjects)}")
print(f"Viability labels: {sum(viability_map.values())} viable, "
      f"{len(viability_map)-sum(viability_map.values())} non-viable")

# ── Channel names ─────────────────────────────────────────────────────────────
print("\nLoading channel names...", end=" ", flush=True)
ch_names = get_eeg_channel_names(subjects[0])
print(f"{len(ch_names)} EEG channels  (first 5: {ch_names[:5]})")

# ── Step 1: Load all R02 data ─────────────────────────────────────────────────
print(f"\n[Step 1] Loading R02 (eyes-closed) for {len(subjects)} subjects...")
t0 = time.time()

all_data_ec = {}
for i, subj in enumerate(subjects, 1):
    print(f"  [{i:2d}/{len(subjects)}] {subj}...", end=" ", flush=True)
    try:
        d = load_r02(subj)
        assert d.shape == (64, 9600), f"Shape {d.shape}"
        all_data_ec[subj] = d
        print("OK")
    except Exception as e:
        print(f"ERROR: {e}")

print(f"Loaded {len(all_data_ec)}/{len(subjects)} subjects  ({(time.time()-t0)/60:.1f} min)")

# ── Step 2: Fit or load microstate templates ──────────────────────────────────
tmpl_path      = RESULTS / "microstate_templates.npy"
tmpl_meta_path = RESULTS / "microstate_templates_meta.json"

if tmpl_path.exists() and tmpl_meta_path.exists():
    print(f"\n[Step 2] Loading cached microstate templates...")
    templates = np.load(tmpl_path)
    with open(tmpl_meta_path) as f:
        tmpl_meta_loaded = json.load(f)
    gev = tmpl_meta_loaded.get("GEV", float("nan"))
    print(f"  Shape: {templates.shape}  GEV = {gev:.4f}")
else:
    print(f"\n[Step 2] Fitting population microstate templates (K=4, 50 inits)...")
    t1 = time.time()
    kmeans    = fit_microstate_templates(all_data_ec, ch_names)
    templates = get_template_maps(kmeans)
    gev       = float(kmeans.GEV_) if hasattr(kmeans, "GEV_") else float("nan")
    print(f"  Done in {(time.time()-t1)/60:.1f} min  GEV = {gev:.4f}")

    np.save(tmpl_path, templates)
    with open(tmpl_meta_path, "w") as f:
        json.dump({
            "n_states": int(templates.shape[0]),
            "n_channels": int(templates.shape[1]),
            "GEV": gev,
            "n_subjects_pooled": len(all_data_ec),
            "channel_names": ch_names,
        }, f, indent=2)
    print(f"  Saved -> {tmpl_path}")

# ── Step 3: Load R01 (eyes-open) ─────────────────────────────────────────────
print(f"\n[Step 3] Loading R01 (eyes-open) for {len(subjects)} subjects...")
all_data_eo = {}
eo_errors   = []
for i, subj in enumerate(subjects, 1):
    print(f"  [{i:2d}/{len(subjects)}] {subj}...", end=" ", flush=True)
    if subj not in all_data_ec:
        print("SKIP (R02 failed)")
        continue
    try:
        d = load_r01(subj)
        assert d.shape == (64, 9600), f"Shape {d.shape}"
        all_data_eo[subj] = d
        print("OK")
    except Exception as e:
        print(f"ERROR: {e}")
        eo_errors.append(subj)

print(f"Loaded {len(all_data_eo)}/{len(subjects)} R01 recordings")
if eo_errors:
    print(f"R01 failed for: {eo_errors}")

# ── Step 4: Extract features for each subject ─────────────────────────────────
print(f"\n[Step 4] Extracting features per subject...")
t2 = time.time()

good_subjects = [s for s in subjects if s in all_data_ec]

X_ms_list  = []
X_sp_list  = []
X_eo_list  = []
y_list     = []
subj_order = []
apf_flagged = []
feat_errors = []

for i, subj in enumerate(good_subjects, 1):
    print(f"  [{i:3d}/{len(good_subjects)}] {subj}...", end=" ", flush=True)
    try:
        data_ec = all_data_ec[subj]

        # Microstate features (26)
        ms_feats = compute_microstate_features(data_ec, templates)

        # Spectral features from EC (47)
        sp_feats, flagged = compute_spectral_features(data_ec, ch_names)

        # EO features + reactivity (42) — if R01 available
        if subj in all_data_eo:
            eo_feats = compute_eo_features(all_data_eo[subj], data_ec, ch_names)
        else:
            eo_feats = np.full(42, np.nan)

        X_ms_list.append(ms_feats)
        X_sp_list.append(sp_feats)
        X_eo_list.append(eo_feats)
        y_list.append(viability_map[subj])
        subj_order.append(subj)

        if flagged:
            apf_flagged.append(subj)

        cov_sum = ms_feats[8:12].sum()
        note = "  [APF edge]" if flagged else ""
        print(f"OK  cov={cov_sum:.4f}{note}")

    except Exception as e:
        print(f"ERROR: {e}")
        feat_errors.append((subj, str(e)))

print(f"\nFeatures done in {(time.time()-t2)/60:.1f} min")
if feat_errors:
    print(f"Errors: {feat_errors}")

# ── Step 5: Impute APF edge cases ─────────────────────────────────────────────
X_ms = np.array(X_ms_list)
X_sp = np.array(X_sp_list)
X_eo = np.array(X_eo_list)
y    = np.array(y_list)

from src.spectral_features import FEATURE_NAMES_SPECTRAL
apf_col  = FEATURE_NAMES_SPECTRAL.index("alpha_peak_freq_cz")
apf_vals = X_sp[:, apf_col]
valid_apf = apf_vals[~np.isnan(apf_vals)]
cohort_median_apf = float(np.median(valid_apf)) if len(valid_apf) > 0 else 10.0

n_nan = int(np.isnan(apf_vals).sum())
if n_nan > 0:
    X_sp[np.isnan(apf_vals), apf_col] = cohort_median_apf
    print(f"Imputed {n_nan} edge APF values with cohort median = {cohort_median_apf:.2f} Hz")

# Impute any NaN EO features (subjects without R01)
eo_has_nan = np.isnan(X_eo).any(axis=1)
if eo_has_nan.any():
    col_medians = np.nanmedian(X_eo, axis=0)
    for j in range(X_eo.shape[1]):
        nan_rows = np.isnan(X_eo[:, j])
        X_eo[nan_rows, j] = col_medians[j]
    print(f"Imputed EO features for {eo_has_nan.sum()} subjects missing R01")

# ── Sanity checks ─────────────────────────────────────────────────────────────
print("\n[Sanity checks]")
cov_sums    = X_ms[:, 8:12].sum(axis=1)
print(f"  Coverage sum max deviation: {np.abs(cov_sums - 1.0).max():.6f}")
print(f"  X_ms NaN: {np.isnan(X_ms).any()}  Inf: {np.isinf(X_ms).any()}")
print(f"  X_sp NaN: {np.isnan(X_sp).any()}  Inf: {np.isinf(X_sp).any()}")
print(f"  X_eo NaN: {np.isnan(X_eo).any()}  Inf: {np.isinf(X_eo).any()}")
print(f"  Shapes: X_ms={X_ms.shape}  X_sp={X_sp.shape}  X_eo={X_eo.shape}")
print(f"  y: viable={int(y.sum())}  non-viable={int((1-y).sum())}")

# ── Save ───────────────────────────────────────────────────────────────────────
out_path = RESULTS / "features.npz"
np.savez(
    out_path,
    subjects=np.array(subj_order),
    X_ms=X_ms,
    X_sp=X_sp,
    X_eo=X_eo,
    y=y,
    feature_names_ms=np.array(FEATURE_NAMES_MICROSTATE),
    feature_names_sp=np.array(FEATURE_NAMES_SPECTRAL),
    feature_names_eo=np.array(FEATURE_NAMES_EO),
    apf_flagged_subjects=np.array(apf_flagged),
    cohort_median_apf=np.array([cohort_median_apf]),
    microstate_templates=templates,
    channel_names=np.array(ch_names),
    microstate_GEV=np.array([gev]),
)
print(f"\nSaved -> {out_path}")
print(f"Total runtime: {(time.time()-t0)/60:.1f} minutes")
