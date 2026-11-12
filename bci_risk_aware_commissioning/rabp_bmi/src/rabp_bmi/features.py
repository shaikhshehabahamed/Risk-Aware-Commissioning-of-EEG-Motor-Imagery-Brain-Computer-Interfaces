from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

from rabp_bmi.dataset import read_raw_edf, subject_run_path


MOTOR_CHANNELS = frozenset(
    {
        "FC5",
        "FC3",
        "FC1",
        "FCZ",
        "FC2",
        "FC4",
        "FC6",
        "C5",
        "C3",
        "C1",
        "CZ",
        "C2",
        "C4",
        "C6",
        "CP5",
        "CP3",
        "CP1",
        "CPZ",
        "CP2",
        "CP4",
        "CP6",
    }
)


@dataclass(frozen=True)
class SubjectFeatures:
    subject: str
    x: np.ndarray
    y: np.ndarray
    runs: np.ndarray
    channel_names: tuple[str, ...]


def clean_channel_name(name: str) -> str:
    return name.replace(".", "").upper()


def channel_indices(raw: mne.io.BaseRaw, channel_set: str) -> list[int]:
    if channel_set == "all":
        return list(range(len(raw.ch_names)))
    if channel_set != "motor":
        raise ValueError(f"Unknown channel_set: {channel_set}")

    indices = [
        idx
        for idx, name in enumerate(raw.ch_names)
        if clean_channel_name(name) in MOTOR_CHANNELS
    ]
    if len(indices) != len(MOTOR_CHANNELS):
        found = {clean_channel_name(raw.ch_names[idx]) for idx in indices}
        missing = sorted(MOTOR_CHANNELS - found)
        raise ValueError(f"Missing expected motor channels: {missing}")
    return indices


def annotation_events_and_labels(
    raw: mne.io.BaseRaw,
    label_mode: str,
) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    if label_mode == "left_right":
        event_id = {"T1": 1, "T2": 2}
        events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose="ERROR")
        labels = (events[:, -1] == 2).astype(int)
        return events, {"left": 1, "right": 2}, labels
    if label_mode == "intent_rest":
        event_id = {"T0": 1, "T1": 2, "T2": 3}
        events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose="ERROR")
        labels = (events[:, -1] != 1).astype(int)
        return events, {"rest": 1, "intent_left": 2, "intent_right": 3}, labels
    raise ValueError(f"Unknown label_mode: {label_mode}")


def extract_run_logvar_features(
    edf_path: Path,
    run: int,
    channel_set: str = "motor",
    l_freq: float = 8.0,
    h_freq: float = 30.0,
    tmin: float = 0.5,
    tmax: float = 3.5,
    label_mode: str = "left_right",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    raw = read_raw_edf(edf_path, preload=True)
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose="ERROR")

    picks = channel_indices(raw, channel_set)
    channel_names = tuple(clean_channel_name(raw.ch_names[idx]) for idx in picks)
    raw.pick(picks, verbose="ERROR")

    events, event_id, labels = annotation_events_and_labels(raw, label_mode)
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose="ERROR",
    )

    data = epochs.get_data(copy=True)
    x = np.log(np.var(data, axis=2) + 1e-12)
    y = labels[epochs.selection]
    runs = np.full(shape=len(y), fill_value=run, dtype=int)
    return x, y, runs, channel_names


def extract_subject_logvar_features(
    dataset_root: Path,
    subject: str,
    task_runs: tuple[int, ...] = (4, 8, 12),
    channel_set: str = "motor",
    label_mode: str = "left_right",
    tmin: float = 0.5,
    tmax: float = 3.5,
) -> SubjectFeatures:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    run_ids: list[np.ndarray] = []
    channel_names: tuple[str, ...] | None = None

    for run in task_runs:
        edf_path = subject_run_path(dataset_root, subject, run)
        x_run, y_run, runs_run, names_run = extract_run_logvar_features(
            edf_path=edf_path,
            run=run,
            channel_set=channel_set,
            label_mode=label_mode,
            tmin=tmin,
            tmax=tmax,
        )
        if channel_names is None:
            channel_names = names_run
        elif channel_names != names_run:
            raise ValueError(f"Channel mismatch for {subject} run {run}")
        xs.append(x_run)
        ys.append(y_run)
        run_ids.append(runs_run)

    if channel_names is None:
        raise ValueError(f"No features extracted for {subject}")

    return SubjectFeatures(
        subject=subject,
        x=np.vstack(xs),
        y=np.concatenate(ys),
        runs=np.concatenate(run_ids),
        channel_names=channel_names,
    )
