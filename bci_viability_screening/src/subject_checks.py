"""
Five programmatic inclusion checks, run in order.
Returns a result dict for each subject; stops at the first failed check.
"""
import numpy as np
import mne

from .config import (
    DATA_RAW, SFREQ, N_CHANNELS,
    RUN_BASELINE, TASK4_RUNS,
    MIN_BASELINE_DURATION, MIN_EPOCHS_PER_CLASS,
    TASK4_BANDPASS, EPOCH_TMIN, EPOCH_TMAX, REJECT_THRESH_V,
    TASK4_EVENT_LABELS, REST_TRIM_START,
)

mne.set_log_level("WARNING")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _edf_path(subject_id: str, run: str):
    return DATA_RAW / subject_id / f"{subject_id}{run}.edf"


def _read_info(subject_id: str, run: str):
    """Load just the file header (no preload). Raises on missing/corrupt file."""
    path = _edf_path(subject_id, run)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found")
    return mne.io.read_raw_edf(str(path), preload=False, verbose=False)


def _load_and_epoch(subject_id: str, run: str):
    """
    Load one Task-4 run, bandpass-filter, epoch, and apply amplitude rejection.
    Returns (n_class0, n_class1) after rejection.
    """
    raw = mne.io.read_raw_edf(str(_edf_path(subject_id, run)),
                               preload=True, verbose=False)

    # 4th-order zero-phase Butterworth bandpass 8-30 Hz
    raw.filter(
        TASK4_BANDPASS[0], TASK4_BANDPASS[1],
        method="iir",
        iir_params=dict(order=4, ftype="butter"),
        verbose=False,
    )

    events, event_id = mne.events_from_annotations(raw, verbose=False)

    # Keep only T1 and T2 annotations
    target_id = {k: v for k, v in event_id.items() if k in TASK4_EVENT_LABELS}
    if not target_id:
        return 0, 0

    epochs = mne.Epochs(
        raw, events,
        event_id=target_id,
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        baseline=None,
        reject={"eeg": REJECT_THRESH_V},
        preload=True,
        verbose=False,
    )
    epochs.drop_bad(verbose=False)

    if len(epochs) == 0:
        return 0, 0

    # Map event codes back to class labels
    rev = {v: k for k, v in target_id.items()}
    labels = np.array([TASK4_EVENT_LABELS[rev[code]] for code in epochs.events[:, 2]])

    n0 = int(np.sum(labels == 0))
    n1 = int(np.sum(labels == 1))
    return n0, n1


# ── Individual checks ──────────────────────────────────────────────────────────

def _check1_sfreq(subject_id: str) -> dict:
    """Check 1: all relevant files have sfreq == 160 Hz."""
    for run in [RUN_BASELINE] + TASK4_RUNS:
        try:
            raw = _read_info(subject_id, run)
        except FileNotFoundError as e:
            return {"pass": False, "reason": f"check1: {e}"}
        except Exception as e:
            return {"pass": False, "reason": f"check1: EDF read error ({run}): {e}"}

        sfreq = raw.info["sfreq"]
        if sfreq != SFREQ:
            return {"pass": False, "reason": f"check1: {run} sfreq={sfreq} (expected {SFREQ})"}

    return {"pass": True, "reason": ""}


def _check2_baseline_duration(subject_id: str) -> dict:
    """Check 2: R02 has >= 50 s of usable signal after skipping the first 1 s."""
    try:
        raw = _read_info(subject_id, RUN_BASELINE)
    except Exception as e:
        return {"pass": False, "reason": f"check2: {e}"}

    usable = raw.times[-1] - REST_TRIM_START
    if usable < MIN_BASELINE_DURATION:
        return {
            "pass": False,
            "reason": f"check2: R02 usable={usable:.1f}s < {MIN_BASELINE_DURATION}s",
        }
    return {"pass": True, "reason": ""}


def _check3_epoch_yield(subject_id: str) -> dict:
    """
    Checks 3 & 4: each of R06, R10, R14 must have >= 5 valid epochs per class
    after 400 µV rejection (check 4 — class presence — is subsumed by check 3).
    Returns epoch_counts alongside pass/fail.
    """
    epoch_counts = {}
    for run in TASK4_RUNS:
        try:
            n0, n1 = _load_and_epoch(subject_id, run)
        except Exception as e:
            return {"pass": False, "reason": f"check3: {run} error: {e}", "epoch_counts": {}}

        epoch_counts[run] = {"class0": n0, "class1": n1}

        if n0 < MIN_EPOCHS_PER_CLASS:
            return {
                "pass": False,
                "reason": (
                    f"check3: {run} class0 has {n0} epochs "
                    f"(need >= {MIN_EPOCHS_PER_CLASS})"
                ),
                "epoch_counts": epoch_counts,
            }
        if n1 < MIN_EPOCHS_PER_CLASS:
            return {
                "pass": False,
                "reason": (
                    f"check3: {run} class1 has {n1} epochs "
                    f"(need >= {MIN_EPOCHS_PER_CLASS})"
                ),
                "epoch_counts": epoch_counts,
            }

    return {"pass": True, "reason": "", "epoch_counts": epoch_counts}


def _check5_channels(subject_id: str) -> dict:
    """Check 5: exactly 64 EEG channels in R02 and all three Task-4 runs."""
    for run in [RUN_BASELINE] + TASK4_RUNS:
        try:
            raw = _read_info(subject_id, run)
        except Exception as e:
            return {"pass": False, "reason": f"check5: {e}"}

        n_eeg = len(mne.pick_types(raw.info, eeg=True))
        if n_eeg != N_CHANNELS:
            return {
                "pass": False,
                "reason": f"check5: {run} has {n_eeg} EEG channels (expected {N_CHANNELS})",
            }

    return {"pass": True, "reason": ""}


# ── Public API ─────────────────────────────────────────────────────────────────

def check_subject(subject_id: str) -> dict:
    """
    Run all five inclusion checks in order; stop at first failure.

    Returns
    -------
    dict with keys:
        subject       : str
        included      : bool
        failed_check  : int | None
        reason        : str
        epoch_counts  : dict  (populated if check3 ran successfully)
    """
    result = {
        "subject": subject_id,
        "included": False,
        "failed_check": None,
        "reason": "",
        "epoch_counts": {},
    }

    for check_fn, check_num in [
        (_check1_sfreq,             1),
        (_check2_baseline_duration, 2),
        (_check3_epoch_yield,       3),
        (_check5_channels,          5),
    ]:
        r = check_fn(subject_id)
        if not r["pass"]:
            result["failed_check"] = check_num
            result["reason"] = r["reason"]
            result["epoch_counts"] = r.get("epoch_counts", {})
            return result
        if check_num == 3:
            result["epoch_counts"] = r.get("epoch_counts", {})

    result["included"] = True
    return result
