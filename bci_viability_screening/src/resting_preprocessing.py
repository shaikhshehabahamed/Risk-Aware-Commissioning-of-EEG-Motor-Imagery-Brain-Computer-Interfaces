"""
Resting-state EEG preprocessing.

Provides loaders for both baselines:
  load_r02()  — eyes-closed (R02): primary microstate + spectral features
  load_r01()  — eyes-open  (R01): used for alpha-reactivity features

Both apply the same pipeline: strip channel dots, pick EEG, trim 1 s,
bandpass 1-40 Hz (4th-order zero-phase Butterworth), average reference.
Output shape: (64, 9600) — 64 channels × 60 s × 160 Hz.
"""

import numpy as np
import mne

from .config import (
    DATA_RAW,
    SFREQ,
    RUN_BASELINE,
    RUN_BASELINE_EO,
    REST_BANDPASS,
    REST_TRIM_START,
    REST_DURATION,
)

mne.set_log_level("WARNING")


def get_eeg_channel_names(subject_id: str = "S001") -> list:
    """
    Return the 64 EEG channel names (trailing dots stripped) from one subject's R02.
    All included subjects share the same montage (check 5 guarantees this).
    """
    fpath = DATA_RAW / subject_id / f"{subject_id}{RUN_BASELINE}.edf"
    raw = mne.io.read_raw_edf(str(fpath), preload=False, verbose=False)
    raw.rename_channels({ch: ch.rstrip(".") for ch in raw.ch_names})
    raw.pick_types(eeg=True)
    return list(raw.ch_names)


def load_r02(subject_id: str) -> np.ndarray:
    """
    Load and preprocess the eyes-closed resting recording (R02).

    Steps
    -----
    1. Read EDF, strip trailing dots from channel names.
    2. Pick only EEG channels.
    3. Skip first REST_TRIM_START seconds; keep exactly REST_DURATION seconds.
       If recording is shorter than (REST_TRIM_START + REST_DURATION), use all
       available data from REST_TRIM_START onward (caller's responsibility to
       check that >= 50 s was available via subject checks).
    4. Bandpass 1-40 Hz (4th-order zero-phase Butterworth).
    5. Apply average reference.

    Returns
    -------
    data : ndarray, shape (64, 9600)
        64 channels × 9600 samples (60 s × 160 Hz), in SI volts.
    """
    fpath = DATA_RAW / subject_id / f"{subject_id}{RUN_BASELINE}.edf"
    raw = mne.io.read_raw_edf(str(fpath), preload=True, verbose=False)

    # Strip trailing dots from BCI2000 channel names (e.g. 'Fc5.' -> 'Fc5')
    raw.rename_channels({ch: ch.rstrip(".") for ch in raw.ch_names})

    # Ensure only EEG channels
    raw.pick_types(eeg=True)

    # ── Time window ───────────────────────────────────────────────────────────
    n_skip  = int(REST_TRIM_START * SFREQ)                   # samples to skip
    n_use   = int(REST_DURATION   * SFREQ)                   # samples to use
    total_samples = raw.n_times

    start = n_skip
    stop  = min(start + n_use, total_samples)                # guard short files
    data  = raw.get_data(start=start, stop=stop)             # (64, n_samples)

    # If shorter than 60 s (shouldn't happen after check 2, but be safe)
    if data.shape[1] < n_use:
        pad = np.zeros((data.shape[0], n_use - data.shape[1]))
        data = np.concatenate([data, pad], axis=1)

    # ── Bandpass 1-40 Hz on the extracted array ───────────────────────────────
    # Use scipy directly on the numpy array to avoid MNE overhead
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, [REST_BANDPASS[0], REST_BANDPASS[1]],
                 btype="bandpass", fs=SFREQ, output="sos")
    data = sosfiltfilt(sos, data, axis=1)

    # ── Average reference ─────────────────────────────────────────────────────
    data -= data.mean(axis=0, keepdims=True)   # subtract mean across channels at each sample

    return data.astype(np.float64)


def _load_baseline_run(subject_id: str, run_id: str) -> np.ndarray:
    """Shared loader for any baseline run (R01 or R02)."""
    fpath = DATA_RAW / subject_id / f"{subject_id}{run_id}.edf"
    raw   = mne.io.read_raw_edf(str(fpath), preload=True, verbose=False)
    raw.rename_channels({ch: ch.rstrip(".") for ch in raw.ch_names})
    raw.pick_types(eeg=True)

    n_skip = int(REST_TRIM_START * SFREQ)
    n_use  = int(REST_DURATION   * SFREQ)
    start  = n_skip
    stop   = min(start + n_use, raw.n_times)
    data   = raw.get_data(start=start, stop=stop)

    if data.shape[1] < n_use:
        pad  = np.zeros((data.shape[0], n_use - data.shape[1]))
        data = np.concatenate([data, pad], axis=1)

    from scipy.signal import butter, sosfiltfilt
    sos  = butter(4, [REST_BANDPASS[0], REST_BANDPASS[1]],
                  btype="bandpass", fs=SFREQ, output="sos")
    data = sosfiltfilt(sos, data, axis=1)
    data -= data.mean(axis=0, keepdims=True)
    return data.astype(np.float64)


def load_r01(subject_id: str) -> np.ndarray:
    """
    Load and preprocess the eyes-open resting recording (R01).
    Returns ndarray (64, 9600).
    """
    return _load_baseline_run(subject_id, RUN_BASELINE_EO)
