"""
Generic FBCSP-LDA pipeline for any 3-run motor-imagery task,
with leave-one-run-out (LORO) cross-validation.

Works identically for:
  Task 2 (R04/R08/R12): left vs right hand MI  [primary study target]
  Task 4 (R06/R10/R14): bilateral MI            [used in subject checks only]

Fold design (LORO, prospective):
  fold1            : train runs[1]+runs[2], test runs[0]
  fold2            : train runs[0]+runs[2], test runs[1]
  fold3_prospective: train runs[0]+runs[1], test runs[2]  <- viability label

Viability = Fold-3 accuracy >= VIABILITY_THRESHOLD (0.60).
"""

import numpy as np
import mne
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .config import (
    DATA_RAW, SFREQ,
    TASK_BANDPASS, EPOCH_TMIN, EPOCH_TMAX, REJECT_THRESH_V,
    TASK_EVENT_LABELS,
    FBCSP_SUBBANDS, CSP_N_COMPONENTS,
    VIABILITY_THRESHOLD,
)

mne.set_log_level("WARNING")


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_run_data(subject_id: str, run: str) -> dict:
    """
    Load one run and return per-sub-band epoch arrays.

    Strategy: broad-band (8-30 Hz) copy used only for artifact rejection;
    sub-band filtering applied to the ORIGINAL continuous raw to avoid edge
    effects from filtering short (~3.5 s) epochs.

    Returns
    -------
    {X_per_band: {(lo,hi): (n_kept, 64, n_times)}, y, n_original, n_kept}
    """
    fpath = DATA_RAW / subject_id / f"{subject_id}{run}.edf"
    raw   = mne.io.read_raw_edf(str(fpath), preload=True, verbose=False)

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    target_id = {k: v for k, v in event_id.items() if k in TASK_EVENT_LABELS}
    if not target_id:
        raise RuntimeError(f"{subject_id}/{run}: no T1/T2 annotations found")

    n_original = int(np.sum(np.isin(events[:, 2], list(target_id.values()))))

    # ── Broad-band copy for artifact rejection only ───────────────────────────
    raw_broad = raw.copy()
    raw_broad.filter(
        TASK_BANDPASS[0], TASK_BANDPASS[1],
        method="iir", iir_params=dict(order=4, ftype="butter"),
        verbose=False,
    )
    epochs_broad = mne.Epochs(
        raw_broad, events, event_id=target_id,
        tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
        baseline=None, reject={"eeg": REJECT_THRESH_V},
        preload=True, verbose=False,
    )
    epochs_broad.drop_bad(verbose=False)
    n_kept = len(epochs_broad)

    if n_kept == 0:
        return {"X_per_band": {}, "y": np.array([]),
                "n_original": n_original, "n_kept": 0}

    rev = {v: k for k, v in target_id.items()}
    y   = np.array([TASK_EVENT_LABELS[rev[c]] for c in epochs_broad.events[:, 2]])
    survived_events = epochs_broad.events

    # ── Sub-band epochs from original (unfiltered) raw ────────────────────────
    X_per_band = {}
    for (lo, hi) in FBCSP_SUBBANDS:
        raw_sub = raw.copy()
        raw_sub.filter(lo, hi, method="iir",
                       iir_params=dict(order=4, ftype="butter"), verbose=False)
        epochs_sub = mne.Epochs(
            raw_sub, survived_events, event_id=target_id,
            tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
            baseline=None, reject=None,       # rejection already done above
            preload=True, verbose=False,
        )
        X_per_band[(lo, hi)] = epochs_sub.get_data()

    return {"X_per_band": X_per_band, "y": y,
            "n_original": n_original, "n_kept": n_kept}


# ── FBCSP ──────────────────────────────────────────────────────────────────────

def _fbcsp_fit_transform(X_train_pb, y_train, X_test_pb):
    """Fit CSP per sub-band on training; transform both splits."""
    F_train, F_test = [], []
    for band in FBCSP_SUBBANDS:
        csp = CSP(n_components=CSP_N_COMPONENTS, reg="ledoit_wolf", log=True)
        csp.fit(X_train_pb[band], y_train)
        F_train.append(csp.transform(X_train_pb[band]))
        F_test.append(csp.transform(X_test_pb[band]))
    return np.hstack(F_train), np.hstack(F_test)


def _concat_per_band(run_data, runs):
    return {b: np.concatenate([run_data[r]["X_per_band"][b] for r in runs])
            for b in FBCSP_SUBBANDS}

def _concat_y(run_data, runs):
    return np.concatenate([run_data[r]["y"] for r in runs])


# ── LORO-CV ────────────────────────────────────────────────────────────────────

def _run_loro_cv(run_data: dict, runs: list) -> dict:
    """
    3-fold LORO-CV.  runs = [r0, r1, r2]; fold3 is prospective (test = r2).
    """
    r0, r1, r2 = runs
    folds_cfg = [
        ("fold1",             [r1, r2], r0),
        ("fold2",             [r0, r2], r1),
        ("fold3_prospective", [r0, r1], r2),
    ]
    fold_results = {}
    for name, train_runs, test_run in folds_cfg:
        X_tr  = _concat_per_band(run_data, train_runs)
        y_tr  = _concat_y(run_data, train_runs)
        X_te  = run_data[test_run]["X_per_band"]
        y_te  = run_data[test_run]["y"]

        F_tr, F_te = _fbcsp_fit_transform(X_tr, y_tr, X_te)

        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(F_tr, y_tr)
        acc = float(lda.score(F_te, y_te))

        fold_results[name] = {
            "accuracy": acc,
            "n_train":  int(len(y_tr)),
            "n_test":   int(len(y_te)),
        }

    fold3_acc = fold_results["fold3_prospective"]["accuracy"]
    mean_acc  = float(np.mean([v["accuracy"] for v in fold_results.values()]))
    return {
        "folds":          fold_results,
        "mean_loro_acc":  mean_acc,
        "fold3_accuracy": fold3_acc,
        "viable":         bool(fold3_acc >= VIABILITY_THRESHOLD),
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def compute_viability(subject_id: str, task_runs: list) -> dict:
    """
    Load MI runs, run FBCSP-LDA LORO-CV, return viability dict.

    Parameters
    ----------
    subject_id : e.g. 'S001'
    task_runs  : list of 3 run IDs, e.g. ['R04','R08','R12']

    Returns
    -------
    dict: subject, viable, fold3_accuracy, mean_loro_acc, folds,
          epoch_counts, error
    """
    result = {
        "subject":        subject_id,
        "viable":         None,
        "fold3_accuracy": None,
        "mean_loro_acc":  None,
        "folds":          {},
        "epoch_counts":   {},
        "error":          None,
    }
    try:
        run_data = {}
        for run in task_runs:
            rd = _load_run_data(subject_id, run)
            run_data[run] = rd
            result["epoch_counts"][run] = {
                "n_original": rd["n_original"],
                "n_kept":     rd["n_kept"],
            }
        cv = _run_loro_cv(run_data, task_runs)
        result.update({
            "viable":         cv["viable"],
            "fold3_accuracy": cv["fold3_accuracy"],
            "mean_loro_acc":  cv["mean_loro_acc"],
            "folds":          cv["folds"],
        })
    except Exception as e:
        result["error"] = str(e)
    return result
