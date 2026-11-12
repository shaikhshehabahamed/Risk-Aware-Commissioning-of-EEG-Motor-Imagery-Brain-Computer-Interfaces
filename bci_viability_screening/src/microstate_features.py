"""
EEG microstate feature extraction.

Population-level pipeline:
  1. Concatenate all included subjects' preprocessed R02 data into one recording.
  2. Fit polarity-invariant modified k-means (K=4) via pycrostates.ModKMeans.
     pycrostates internally extracts GFP peaks and fits on those topographies.
  3. Extract cluster-centre templates (n_states, n_channels).
  4. Back-fit templates to each subject's continuous signal using absolute correlation.
  5. Apply temporal smoothing (reassign runs < 30 ms to surrounding window dominant).
  6. Compute 26-feature vector per subject:
       - 12 temporal statistics  (4 states × {mean duration, occurrence, coverage})
       - 12 transition probs     (4×4 off-diagonal row-normalised matrix)
       -  2 GFP global stats     (mean GFP, std GFP)

Reference: Pascual-Marqui et al. (1995) modified k-means with polarity invariance.
"""

import numpy as np
import mne
from scipy.signal import argrelmax
from pycrostates.cluster import ModKMeans

from .config import (
    SFREQ,
    N_MICROSTATES,
    MICROSTATE_N_RUNS,
    SMOOTHING_MIN_SAMPLES,
)

mne.set_log_level("WARNING")

FEATURE_NAMES_MICROSTATE = (
    [f"ms_dur_{k}"  for k in range(4)]
    + [f"ms_occ_{k}"  for k in range(4)]
    + [f"ms_cov_{k}"  for k in range(4)]
    + [f"ms_T{i}{j}" for i in range(4) for j in range(4) if i != j]
    + ["gfp_mean", "gfp_std"]
)
assert len(FEATURE_NAMES_MICROSTATE) == 26


# ── GFP ───────────────────────────────────────────────────────────────────────

def compute_gfp(data: np.ndarray) -> np.ndarray:
    """Std across channels at each sample. data: (n_channels, n_times) -> (n_times,)"""
    return data.std(axis=0)


# ── Population template fitting ────────────────────────────────────────────────

def fit_microstate_templates(
    all_data: dict,
    ch_names: list,
    n_states: int = N_MICROSTATES,
    n_runs: int = MICROSTATE_N_RUNS,
    random_state: int = 42,
) -> "ModKMeans":
    """
    Fit polarity-invariant k-means on pooled resting-state data from all subjects.

    pycrostates internally computes GFP peaks and fits on those topographies,
    which is the standard population-level microstate fitting procedure.

    Parameters
    ----------
    all_data     : {subject_id: ndarray (n_channels, n_times)}
    ch_names     : channel names after dot-stripping (length = n_channels)
    n_states     : K (default 4)
    n_runs       : random initialisations (default 200)
    random_state : for reproducibility

    Returns
    -------
    Fitted ModKMeans object.
    """
    # Concatenate all subjects' data into one long recording
    combined = np.concatenate(list(all_data.values()), axis=1)  # (64, 92*9600)

    info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types="eeg")
    raw_pool = mne.io.RawArray(combined, info, verbose=False)

    kmeans = ModKMeans(
        n_clusters=n_states,
        n_init=n_runs,
        random_state=random_state,
    )
    kmeans.fit(raw_pool, picks="eeg", verbose=False)
    return kmeans


def get_template_maps(kmeans: "ModKMeans") -> np.ndarray:
    """Extract cluster-centre maps. Returns (n_states, n_channels)."""
    return kmeans.cluster_centers_


# ── Back-fitting ───────────────────────────────────────────────────────────────

def backfit_templates(data: np.ndarray, templates: np.ndarray) -> np.ndarray:
    """
    Assign each sample to the most correlated template (polarity-invariant).

    Parameters
    ----------
    data      : (n_channels, n_times)
    templates : (n_states, n_channels)

    Returns
    -------
    states : (n_times,) int array in {0, ..., n_states-1}
    """
    # Unit-norm normalise each time point and each template
    eps = 1e-10
    data_norm = data  / (np.linalg.norm(data,      axis=0, keepdims=True) + eps)
    tmpl_norm = templates / (np.linalg.norm(templates, axis=1, keepdims=True) + eps)

    sim    = np.abs(tmpl_norm @ data_norm)   # (n_states, n_times)
    states = np.argmax(sim, axis=0)          # (n_times,)
    return states.astype(np.int32)


# ── Temporal smoothing ─────────────────────────────────────────────────────────

def smooth_states(states: np.ndarray, min_samples: int = SMOOTHING_MIN_SAMPLES) -> np.ndarray:
    """
    Reassign any contiguous run shorter than min_samples to the dominant state
    of the surrounding 100-ms window (16 samples at 160 Hz on each side).

    Parameters
    ----------
    states      : (n_times,) integer state sequence
    min_samples : minimum run length; runs below this are reassigned (default 5)

    Returns
    -------
    (n_times,) smoothed state sequence
    """
    states  = states.copy()
    n       = len(states)
    w_half  = 8   # 8 samples on each side = 100 ms at 160 Hz

    i = 0
    while i < n:
        # Find end of current run
        j = i + 1
        while j < n and states[j] == states[i]:
            j += 1

        if (j - i) < min_samples:
            lo   = max(0, i - w_half)
            hi   = min(n, j + w_half)
            surr = np.concatenate([states[lo:i], states[j:hi]])
            if len(surr) > 0:
                dominant   = np.bincount(surr, minlength=4).argmax()
                states[i:j] = dominant

        i = j

    return states


# ── 26-feature vector ──────────────────────────────────────────────────────────

def compute_microstate_features(
    data: np.ndarray, templates: np.ndarray
) -> np.ndarray:
    """
    Compute the 26-feature microstate vector for one subject.

    Parameters
    ----------
    data      : (n_channels, n_times) — preprocessed resting EEG (avg-ref, 1-40 Hz)
    templates : (n_states, n_channels) — population cluster centres

    Returns
    -------
    features : (26,) float64 array
        [d0..d3, occ0..occ3, cov0..cov3,
         T01,T02,T03, T10,T12,T13, T20,T21,T23, T30,T31,T32,
         GFP_mean, GFP_std]
    """
    n_states = templates.shape[0]
    n_times  = data.shape[1]
    dur_s    = n_times / SFREQ      # recording duration in seconds

    gfp    = compute_gfp(data)
    states = backfit_templates(data, templates)
    states = smooth_states(states)

    # ── Per-state temporal statistics ─────────────────────────────────────────
    mean_durations = np.zeros(n_states)
    occurrences    = np.zeros(n_states)
    coverages      = np.zeros(n_states)

    for k in range(n_states):
        mask          = (states == k)
        total_samples = int(mask.sum())
        coverages[k]  = total_samples / n_times

        if total_samples == 0:
            continue

        # Number of distinct contiguous runs of state k
        diff       = np.diff(mask.astype(np.int8))
        n_segments = int((diff == 1).sum())
        if mask[0]:          # state starts at sample 0
            n_segments += 1

        occurrences[k] = n_segments / dur_s

        if n_segments > 0:
            mean_durations[k] = (total_samples / n_segments) * (1000.0 / SFREQ)  # ms

    # ── Transition probability matrix (12 off-diagonal entries) ───────────────
    transitions = np.zeros((n_states, n_states), dtype=np.int64)
    for t in range(1, n_times):
        if states[t] != states[t - 1]:
            transitions[states[t - 1], states[t]] += 1

    row_sums = transitions.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1.0   # avoid divide-by-zero for absent states
    T = transitions / row_sums       # row-normalised; diagonal is identically 0

    off_diag = np.array([T[i, j] for i in range(n_states)
                                   for j in range(n_states) if i != j])  # 12 values

    # ── GFP global stats ──────────────────────────────────────────────────────
    gfp_feats = np.array([gfp.mean(), gfp.std()])

    # ── Assemble 26-feature vector ────────────────────────────────────────────
    features = np.concatenate([mean_durations, occurrences, coverages, off_diag, gfp_feats])
    assert len(features) == 26, f"Expected 26, got {len(features)}"
    return features.astype(np.float64)
