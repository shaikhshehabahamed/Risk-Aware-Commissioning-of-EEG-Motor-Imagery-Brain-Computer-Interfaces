from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PROJECT_SRC))

from rabp_bmi.config import load_config
from rabp_bmi.dataset import discover_subjects, discover_task_runs, event_counts, read_raw_edf


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit local EEGMMIDB dataset structure.")
    parser.add_argument("--config", default="configs/local.json")
    parser.add_argument("--max-read", type=int, default=12)
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    subjects = discover_subjects(cfg.dataset_root)
    task_files = discover_task_runs(cfg.dataset_root, cfg.task_runs)

    out_dir = cfg.run_root / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "dataset_audit.csv"

    rows: list[dict[str, str | int | float]] = []
    read_errors: list[str] = []

    for idx, run_file in enumerate(task_files):
        row: dict[str, str | int | float] = {
            "subject": run_file.subject,
            "run": run_file.run,
            "edf_exists": int(run_file.edf_path.exists()),
            "event_exists": int(run_file.event_path is not None),
            "file_size_bytes": run_file.edf_path.stat().st_size if run_file.edf_path.exists() else 0,
            "read_checked": 0,
        }
        if idx < args.max_read:
            try:
                raw = read_raw_edf(run_file.edf_path, preload=False)
                counts = event_counts(raw)
                row.update(
                    {
                        "read_checked": 1,
                        "n_channels": len(raw.ch_names),
                        "sfreq": float(raw.info["sfreq"]),
                        "duration_sec": float(raw.times[-1]),
                        "ann_T0": counts.get("T0", 0),
                        "ann_T1": counts.get("T1", 0),
                        "ann_T2": counts.get("T2", 0),
                    }
                )
            except Exception as exc:
                read_errors.append(f"{run_file.edf_path}: {exc}")
                row["read_error"] = str(exc)
        rows.append(row)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"dataset_root={cfg.dataset_root}")
    print(f"subjects={len(subjects)}")
    print(f"task_runs={list(cfg.task_runs)}")
    print(f"task_edf_files={len(task_files)}")
    print(f"read_checked={min(args.max_read, len(task_files))}")
    print(f"audit_csv={summary_path}")
    if read_errors:
        print("read_errors:")
        for error in read_errors:
            print(f"  {error}")


if __name__ == "__main__":
    main()

