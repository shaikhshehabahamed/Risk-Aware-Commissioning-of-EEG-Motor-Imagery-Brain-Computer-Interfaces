"""
Spectral feature extraction from resting-state EEG.

Features computed from R02 (eyes-closed, 60 s):
  [A] Band powers: 6 channels (C3/Cz/C4/CP3/CPz/CP4) x 6 bands     = 36
  [B] Lateralisation indices (C4 vs C3):                             =  3
        alpha (8-12 Hz), SMR (12-15 Hz), beta (15-30 Hz)
  [C] Centroparietal lateralisation (CP4 vs CP3):                    =  3
        alpha, SMR, beta
  [D] C3-C4 inter-hemispheric coherence:                             =  2
        alpha (8-12 Hz), beta (12-30 Hz)
  [E] Individual alpha peak frequency at Cz (parabolic interpolation)=  1
  [F] Spectral entropy (Shannon, 1-40 Hz, all channels)              =  1
  [G] Theta/alpha ratio at Cz (frontal arousal proxy)                =  1
                                                               Total = 47

Features computed from R01 (eyes-open, 60 s):
  [H] Band powers: 6 channels x 6 bands                             = 36
  [I] Lateralisation indices (C4 vs C3): alpha, SMR, beta            =  3

Eyes-open/closed alpha reactivity (R02 / R01):
  [J] Alpha reactivity at C3, Cz, C4 (log ratio EC/EO)              =  3

Grand total when EC + EO features used: 47 + 36 + 3 + 3 = 89 features.
When only EC used (R01 unavailable): 47 features.
"""

import numpy as np
from scipy.signal import welch, coherence

from .config import (
    SFREQ,
    SPECTRAL_CHANNELS,
    SPECTRAL_CHANNELS_PRIMARY,
    SPECTRAL_CHANNELS_CP,
    SPECTRAL_BANDS,
    ALPHA_PEAK_RANGE,
    WELCH_WINDOW_S,
    WELCH_OVERLAP,
    REST_BANDPASS,
)


# ── Welch PSD ─────────────────────────────────────────────────────────────────

def compute_psd(data: np.ndarray, ch_names: list) -> tuple:
    """
    Welch PSD for all channels.

    Returns
    -------
    freqs : (n_freqs,)
    psd   : (n_channels, n_freqs)  V^2/Hz
    """
    nperseg  = int(WELCH_WINDOW_S * SFREQ)
    noverlap = int(nperseg * WELCH_OVERLAP)
    freqs, psd = welch(
        data, fs=SFREQ, window="hann",
        nperseg=nperseg, noverlap=noverlap, axis=1,
    )
    return freqs, psd


def _band_power(psd_row: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    """Trapezoidal integration of a single-channel PSD row within [lo, hi] Hz."""
    mask = (freqs >= lo) & (freqs <= hi)
    return float(np.trapz(psd_row[mask], freqs[mask]))


# ── Lateralisation index (log-power ratio) ────────────────────────────────────

def _lateralisation(psd, freqs, idx_ipsi, idx_contra, lo, hi):
    """
    (logP_contra - logP_ipsi) / (|logP_contra| + |logP_ipsi|)

    For right-hand MI: contra = C4 (left motor), ipsi = C3 (right motor).
    Positive value -> right-hemisphere dominant (contra > ipsi), expected for
    right-hand dominance.
    """
    p_i = _band_power(psd[idx_ipsi],  freqs, lo, hi)
    p_c = _band_power(psd[idx_contra], freqs, lo, hi)
    log_i = np.log(p_i + 1e-30)
    log_c = np.log(p_c + 1e-30)
    denom = abs(log_c) + abs(log_i)
    return (log_c - log_i) / denom if denom > 1e-10 else 0.0


# ── Inter-hemispheric coherence ───────────────────────────────────────────────

def _inter_hemispheric_coherence(data: np.ndarray, idx_a: int, idx_b: int,
                                  freqs: np.ndarray, lo: float, hi: float) -> float:
    """
    Mean magnitude-squared coherence (MSC) between two channels in [lo, hi] Hz.
    Uses the same Welch parameters as the PSD.
    """
    nperseg  = int(WELCH_WINDOW_S * SFREQ)
    noverlap = int(nperseg * WELCH_OVERLAP)
    f, coh = coherence(
        data[idx_a], data[idx_b],
        fs=SFREQ, window="hann",
        nperseg=nperseg, noverlap=noverlap,
    )
    mask = (f >= lo) & (f <= hi)
    return float(np.mean(coh[mask]))


# ── Main feature function ─────────────────────────────────────────────────────

def compute_spectral_features(
    data: np.ndarray,
    ch_names: list,
    cohort_median_apf: float = None,
) -> tuple:
    """
    Compute 47 spectral features from one preprocessed resting-state recording.

    Parameters
    ----------
    data              : (n_channels, n_times) — preprocessed EEG (avg-ref, 1-40 Hz)
    ch_names          : channel names after dot-stripping
    cohort_median_apf : used only if APF falls at range edge (rare); pass None
                        for first-pass computation.

    Returns
    -------
    features   : (47,) float64
    apf_flagged: bool — True if APF could not be interpolated (peak at edge)
    """
    freqs, psd = compute_psd(data, ch_names)
    ch_index   = {ch: i for i, ch in enumerate(ch_names)}

    # Check all required channels are present
    all_needed = SPECTRAL_CHANNELS
    missing = [c for c in all_needed if c not in ch_index]
    if missing:
        raise ValueError(f"Missing channels: {missing}")

    idx_c3  = ch_index["C3"]
    idx_cz  = ch_index["Cz"]
    idx_c4  = ch_index["C4"]
    idx_cp3 = ch_index["Cp3"]
    idx_cpz = ch_index["Cpz"]
    idx_cp4 = ch_index["Cp4"]

    # ── [A] Band powers: 6 channels × 6 bands = 36 ───────────────────────────
    band_powers = []
    for ch_idx in [idx_c3, idx_cz, idx_c4, idx_cp3, idx_cpz, idx_cp4]:
        for lo, hi in SPECTRAL_BANDS.values():
            band_powers.append(_band_power(psd[ch_idx], freqs, lo, hi))

    # ── [B] C3-C4 lateralisation: alpha, SMR, beta ───────────────────────────
    lat_c_alpha = _lateralisation(psd, freqs, idx_c3, idx_c4,  8.0, 12.0)
    lat_c_smr   = _lateralisation(psd, freqs, idx_c3, idx_c4, 12.0, 15.0)
    lat_c_beta  = _lateralisation(psd, freqs, idx_c3, idx_c4, 15.0, 30.0)

    # ── [C] CP3-CP4 lateralisation: alpha, SMR, beta ─────────────────────────
    lat_cp_alpha = _lateralisation(psd, freqs, idx_cp3, idx_cp4,  8.0, 12.0)
    lat_cp_smr   = _lateralisation(psd, freqs, idx_cp3, idx_cp4, 12.0, 15.0)
    lat_cp_beta  = _lateralisation(psd, freqs, idx_cp3, idx_cp4, 15.0, 30.0)

    # ── [D] C3-C4 inter-hemispheric coherence: alpha, beta ───────────────────
    coh_alpha = _inter_hemispheric_coherence(data, idx_c3, idx_c4, freqs,  8.0, 12.0)
    coh_beta  = _inter_hemispheric_coherence(data, idx_c3, idx_c4, freqs, 12.0, 30.0)

    # ── [E] Individual alpha peak frequency at Cz (parabolic interpolation) ───
    apf_lo, apf_hi = ALPHA_PEAK_RANGE
    apf_mask     = (freqs >= apf_lo) & (freqs <= apf_hi)
    psd_cz_alpha = psd[idx_cz][apf_mask]
    freqs_alpha  = freqs[apf_mask]

    apf_flagged = False
    peak_bin    = int(np.argmax(psd_cz_alpha))

    if 0 < peak_bin < len(psd_cz_alpha) - 1:
        y0, y1, y2 = (psd_cz_alpha[peak_bin - 1],
                      psd_cz_alpha[peak_bin],
                      psd_cz_alpha[peak_bin + 1])
        denom = y0 - 2 * y1 + y2
        delta = float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5)) if abs(denom) > 1e-30 else 0.0
        df  = freqs_alpha[1] - freqs_alpha[0]
        apf = float(freqs_alpha[peak_bin] + delta * df)
    else:
        apf_flagged = True
        apf = cohort_median_apf if cohort_median_apf is not None else np.nan

    # ── [F] Spectral entropy ──────────────────────────────────────────────────
    full_mask = (freqs >= REST_BANDPASS[0]) & (freqs <= REST_BANDPASS[1])
    mean_psd  = psd[:, full_mask].mean(axis=0)
    norm_psd  = mean_psd / (mean_psd.sum() + 1e-30)
    sp_entropy = float(-np.sum(norm_psd * np.log2(norm_psd + 1e-30)))

    # ── [G] Theta/alpha ratio at Cz ──────────────────────────────────────────
    theta_cz = _band_power(psd[idx_cz], freqs, 4.0, 8.0)
    alpha_cz = _band_power(psd[idx_cz], freqs, 8.0, 12.0)
    theta_alpha_ratio = float(np.log((theta_cz + 1e-30) / (alpha_cz + 1e-30)))

    # ── Assemble 47-feature vector ────────────────────────────────────────────
    features = np.array(
        band_powers                                          # 36
        + [lat_c_alpha, lat_c_smr, lat_c_beta]              #  3
        + [lat_cp_alpha, lat_cp_smr, lat_cp_beta]           #  3
        + [coh_alpha, coh_beta]                             #  2
        + [apf]                                             #  1
        + [sp_entropy]                                      #  1
        + [theta_alpha_ratio],                              #  1
        dtype=np.float64,
    )
    assert len(features) == 47, f"Expected 47 features, got {len(features)}"
    return features, apf_flagged


# ── Eyes-open features ────────────────────────────────────────────────────────

def compute_eo_features(data_eo: np.ndarray, data_ec: np.ndarray,
                        ch_names: list) -> np.ndarray:
    """
    Eyes-open spectral features + alpha reactivity (log EC/EO ratio).

    Parameters
    ----------
    data_eo : (n_channels, n_times) — eyes-open R01, preprocessed
    data_ec : (n_channels, n_times) — eyes-closed R02, same preprocessing
    ch_names: channel names

    Returns
    -------
    (42,) float64:
      36 band powers (EO) + 3 EO lateralisations + 3 alpha reactivity indices
    """
    freqs_eo, psd_eo = compute_psd(data_eo, ch_names)
    freqs_ec, psd_ec = compute_psd(data_ec, ch_names)
    ch_index = {ch: i for i, ch in enumerate(ch_names)}

    idx_c3  = ch_index["C3"]
    idx_cz  = ch_index["Cz"]
    idx_c4  = ch_index["C4"]
    idx_cp3 = ch_index["Cp3"]
    idx_cpz = ch_index["Cpz"]
    idx_cp4 = ch_index["Cp4"]

    # EO band powers (36)
    bp_eo = []
    for ch_idx in [idx_c3, idx_cz, idx_c4, idx_cp3, idx_cpz, idx_cp4]:
        for lo, hi in SPECTRAL_BANDS.values():
            bp_eo.append(_band_power(psd_eo[ch_idx], freqs_eo, lo, hi))

    # EO lateralisation: alpha, SMR, beta (3)
    lat_eo_alpha = _lateralisation(psd_eo, freqs_eo, idx_c3, idx_c4,  8.0, 12.0)
    lat_eo_smr   = _lateralisation(psd_eo, freqs_eo, idx_c3, idx_c4, 12.0, 15.0)
    lat_eo_beta  = _lateralisation(psd_eo, freqs_eo, idx_c3, idx_c4, 15.0, 30.0)

    # Alpha reactivity: log(EC / EO) at C3, Cz, C4 (3)
    # Positive = EC alpha > EO alpha = normal alpha blocking
    def _log_ratio(psd_ec_row, psd_eo_row, f, lo, hi):
        p_ec = _band_power(psd_ec_row, f, lo, hi)
        p_eo = _band_power(psd_eo_row, f, lo, hi)
        return float(np.log((p_ec + 1e-30) / (p_eo + 1e-30)))

    react_c3 = _log_ratio(psd_ec[idx_c3], psd_eo[idx_c3], freqs_ec, 8.0, 12.0)
    react_cz = _log_ratio(psd_ec[idx_cz], psd_eo[idx_cz], freqs_ec, 8.0, 12.0)
    react_c4 = _log_ratio(psd_ec[idx_c4], psd_eo[idx_c4], freqs_ec, 8.0, 12.0)

    features = np.array(
        bp_eo                                           # 36
        + [lat_eo_alpha, lat_eo_smr, lat_eo_beta]      #  3
        + [react_c3, react_cz, react_c4],               #  3
        dtype=np.float64,
    )
    assert len(features) == 42, f"Expected 42, got {len(features)}"
    return features


# ── Feature name lists ────────────────────────────────────────────────────────

_CH6 = ["C3", "Cz", "C4", "CP3", "CPz", "CP4"]
_BANDS = list(SPECTRAL_BANDS.keys())

FEATURE_NAMES_SPECTRAL = (
    [f"bp_ec_{ch}_{b}" for ch in _CH6 for b in _BANDS]            # 36
    + ["lat_c_alpha", "lat_c_smr", "lat_c_beta"]                  #  3
    + ["lat_cp_alpha", "lat_cp_smr", "lat_cp_beta"]               #  3
    + ["coh_c3c4_alpha", "coh_c3c4_beta"]                         #  2
    + ["alpha_peak_freq_cz"]                                       #  1
    + ["spectral_entropy"]                                         #  1
    + ["theta_alpha_ratio_cz"]                                     #  1
)
assert len(FEATURE_NAMES_SPECTRAL) == 47

FEATURE_NAMES_EO = (
    [f"bp_eo_{ch}_{b}" for ch in _CH6 for b in _BANDS]            # 36
    + ["lat_eo_alpha", "lat_eo_smr", "lat_eo_beta"]               #  3
    + ["react_alpha_c3", "react_alpha_cz", "react_alpha_c4"]      #  3
)
assert len(FEATURE_NAMES_EO) == 42


def get_channel_names_from_data(raw_ch_names: list) -> list:
    """Strip trailing dots from BCI2000 channel names."""
    return [ch.rstrip(".") for ch in raw_ch_names]
