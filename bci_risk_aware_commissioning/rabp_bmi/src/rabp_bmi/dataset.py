from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import mne


SUBJECT_PATTERN = re.compile(r"^S(\d{3})$")


@dataclass(frozen=True)
class SubjectRun:
    subject: str
    run: int
    edf_path: Path
    event_path: Path | None


def discover_subjects(dataset_root: Path) -> list[str]:
    subjects: list[str] = []
    for child in dataset_root.iterdir():
        if child.is_dir() and SUBJECT_PATTERN.match(child.name):
            subjects.append(child.name)
    return sorted(subjects)


def subject_run_path(dataset_root: Path, subject: str, run: int) -> Path:
    return dataset_root / subject / f"{subject}R{run:02d}.edf"


def discover_runs(dataset_root: Path, subject: str) -> list[SubjectRun]:
    subject_dir = dataset_root / subject
    runs: list[SubjectRun] = []
    for edf_path in sorted(subject_dir.glob(f"{subject}R*.edf")):
        match = re.search(r"R(\d{2})\.edf$", edf_path.name)
        if match is None:
            continue
        run = int(match.group(1))
        event_path = edf_path.with_suffix(edf_path.suffix + ".event")
        runs.append(
            SubjectRun(
                subject=subject,
                run=run,
                edf_path=edf_path,
                event_path=event_path if event_path.exists() else None,
            )
        )
    return runs


def discover_task_runs(
    dataset_root: Path,
    task_runs: tuple[int, ...] = (4, 8, 12),
) -> list[SubjectRun]:
    found: list[SubjectRun] = []
    for subject in discover_subjects(dataset_root):
        for run in task_runs:
            edf_path = subject_run_path(dataset_root, subject, run)
            event_path = edf_path.with_suffix(edf_path.suffix + ".event")
            if edf_path.exists():
                found.append(
                    SubjectRun(
                        subject=subject,
                        run=run,
                        edf_path=edf_path,
                        event_path=event_path if event_path.exists() else None,
                    )
                )
    return found


def read_raw_edf(edf_path: Path, preload: bool = False) -> mne.io.BaseRaw:
    return mne.io.read_raw_edf(str(edf_path), preload=preload, verbose="ERROR")


def event_counts(raw: mne.io.BaseRaw) -> dict[str, int]:
    counts: dict[str, int] = {}
    for description in raw.annotations.description:
        label = str(description)
        counts[label] = counts.get(label, 0) + 1
    return counts
