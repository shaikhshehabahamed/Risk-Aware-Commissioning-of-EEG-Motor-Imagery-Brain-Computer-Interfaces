"""
FBCSP-LDA pipeline for Task 4 (bilateral MI) with leave-one-run-out cross-validation.

Data flow per run:
  1. Load raw EDF (preloaded).
  2. Apply 8-30 Hz bandpass to a copy → use for amplitude-rejection epoching.
  3. Record which events survived rejection.
  4. For each FBCSP sub-band, filter the ORIGINAL raw to that band, then epoch
     using only the survived events (avoids edge effects from filtering short epochs).
  5. Run LORO-CV: 3 folds, Fold 3 is prospective (train R06+R10, test R14).
  6. Viability = Fold-3 accuracy >= 60%.
"""

import numpy as np
import mne
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .config import (
    DATA_RAW,
    SFREQ,
    TASK4_RUNS,
    TASK4_BANDPASS,
    EPOCH_TMIN,
    EPOCH_TMAX,
    REJECT_THRESH_V,
    TASK4_EVENT_LABELS,
    FBCSP_SUBBANDS,
    CSP_N_COMPONENTS,
    VIABILITY_THRESHOLD,
)

mne.set_log_level("WARNING")


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_run_data(subject_id: str, run: str) -> dict:
    """
    Load one Task-4 run and return per-sub-band epoch arrays.

    Returns
    -------
    dict with:
        X_per_band   : {(lo, hi): ndarray (n_kept, 64, n_times)}
        y            : ndarray (n_kept,) — binary class labels
        n_original   : int — cue events before rejection
        n_kept       : int — epochs after rejection
    """
    fpath = DATA_RAW / subject_id / f"{subject_id}{run}.edf"
    raw = mne.io.read_raw_edf(str(fpath), preload=True, verbose=False)

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    target_id = {k: v for k, v in event_id.items() if k in TASK4_EVENT_LABELS}
    if not target_id:
        raise RuntimeError(f"{subject_id}/{run}: no T1/T2 annotations found")

    n_original = int(np.sum(np.isin(events[:, 2], list(target_id.values()))))

    # ── Step 1: Broad-band copy for rejection ─────────────────────────────────
    raw_broad = raw.copy()
    raw_broad.filter(
        TASK4_BANDPASS[0], TASK4_BANDPASS[1],
        method="iir", iir_params=dict(order=4, ftype="butter"),
        verbose=False,
    )
    epochs_broad = mne.Epochs(
        raw_broad, events, event_id=target_id,
        tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
        baseline=None,
        reject={"eeg": REJECT_THRESH_V},
        preload=True, verbose=False,
    )
    epochs_broad.drop_bad(verbose=False)

    n_kept = len(epochs_broad)
    if n_kept == 0:
        return {"X_per_band": {}, "y": np.array([]), "n_original": n_original, "n_kept": 0}

    # Build label array
    rev = {v: k for k, v in target_id.items()}
    y = np.array([TASK4_EVENT_LABELS[rev[code]] for code in epochs_broad.events[:, 2]])

    # The events that survived rejection — reuse for all sub-band extractions
    survived_events = epochs_broad.events  # shape (n_kept, 3)

    # ── Step 2: Sub-band epochs from original (unfiltered) raw ───────────────
    X_per_band = {}
    for (lo, hi) in FBCSP_SUBBANDS:
        raw_sub = raw.copy()
        raw_sub.filter(
            lo, hi,
            method="iir", iir_params=dict(order=4, ftype="butter"),
            verbose=False,
        )
        epochs_sub = mne.Epochs(
            raw_sub, survived_events, event_id=target_id,
            tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
            baseline=None,
            reject=None,          # rejection was handled via broad-band
            preload=True, verbose=False,
        )
        X_per_band[(lo, hi)] = epochs_sub.get_data()  # (n_kept, 64, n_times)

    return {"X_per_band": X_per_band, "y": y, "n_original": n_original, "n_kept": n_kept}


# ── FBCSP helpers ──────────────────────────────────────────────────────────────

def _fbcsp_fit_transform(
    X_train_pb: dict, y_train: np.ndarray, X_test_pb: dict
) -> tuple:
    """
    Fit one CSP per sub-band on training data; transform both train and test.

    Parameters
    ----------
    X_train_pb : {(lo,hi): (n_train, n_channels, n_times)}
    y_train    : (n_train,)
    X_test_pb  : {(lo,hi): (n_test,  n_channels, n_times)}

    Returns
    -------
    F_train : (n_train, n_subbands * n_components)
    F_test  : (n_test,  n_subbands * n_components)
    """
    F_train_parts, F_test_parts = [], []
    for band in FBCSP_SUBBANDS:
        csp = CSP(n_components=CSP_N_COMPONENTS, reg="ledoit_wolf", log=True)
        csp.fit(X_train_pb[band], y_train)
        F_train_parts.append(csp.transform(X_train_pb[band]))
        F_test_parts.append(csp.transform(X_test_pb[band]))
    return np.hstack(F_train_parts), np.hstack(F_test_parts)


def _concat_runs(run_data: dict, runs: list, key: str) -> np.ndarray:
    """Concatenate a per-band array (or y) across multiple runs."""
    return np.concatenate([run_data[r][key] for r in runs])


def _concat_per_band(run_data: dict, runs: list) -> dict:
    """Concatenate X_per_band dicts across multiple runs."""
    return {
        band: np.concatenate([run_data[r]["X_per_band"][band] for r in runs])
        for band in FBCSP_SUBBANDS
    }


# ── LORO-CV ────────────────────────────────────────────────────────────────────

LORO_FOLDS = [
    ("fold1",               ["R10", "R14"], "R06"),
    ("fold2",               ["R06", "R14"], "R10"),
    ("fold3_prospective",   ["R06", "R10"], "R14"),   # primary viability fold
]


def _run_loro_cv(run_data: dict) -> dict:
    """
    Run 3-fold LORO-CV on pre-loaded per-band epoch data.

    Parameters
    ----------
    run_data : {run: {'X_per_band': {...}, 'y': ...}}

    Returns
    -------
    dict with fold results and viability label.
    """
    fold_results = {}
    for fold_name, train_runs, test_run in LORO_FOLDS:
        X_train_pb = _concat_per_band(run_data, train_runs)
        y_train    = _concat_runs(run_data, train_runs, "y")
        X_test_pb  = run_data[test_run]["X_per_band"]
        y_test     = run_data[test_run]["y"]

        F_train, F_test = _fbcsp_fit_transform(X_train_pb, y_train, X_test_pb)

        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(F_train, y_train)

        acc = float(lda.score(F_test, y_test))
        fold_results[fold_name] = {
            "accuracy":  acc,
            "n_train":   int(len(y_train)),
            "n_test":    int(len(y_test)),
        }

    fold3_acc = fold_results["fold3_prospective"]["accuracy"]
    mean_acc  = float(np.mean([v["accuracy"] for v in fold_results.values()]))

    return {
        "folds":            fold_results,
        "mean_loro_acc":    mean_acc,
        "fold3_accuracy":   fold3_acc,
        "viable":           bool(fold3_acc >= VIABILITY_THRESHOLD),
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def compute_viability(subject_id: str) -> dict:
    """
    Load Task-4 runs, run FBCSP-LDA LORO-CV, return viability result.

    Returns
    -------
    dict with keys: subject, viable, fold3_accuracy, mean_loro_acc,
                    folds, epoch_counts, error (if any)
    """
    result = {
        "subject":          subject_id,
        "viable":           None,
        "fold3_accuracy":   None,
        "mean_loro_acc":    None,
        "folds":            {},
        "epoch_counts":     {},
        "error":            None,
    }
    try:
        run_data = {}
        for run in TASK4_RUNS:
            rd = _load_run_data(subject_id, run)
            run_data[run] = rd
            result["epoch_counts"][run] = {
                "n_original": rd["n_original"],
                "n_kept":     rd["n_kept"],
            }

        cv = _run_loro_cv(run_data)
        result.update({
            "viable":         cv["viable"],
            "fold3_accuracy": cv["fold3_accuracy"],
            "mean_loro_acc":  cv["mean_loro_acc"],
            "folds":          cv["folds"],
        })
    except Exception as e:
        result["error"] = str(e)

    return result
