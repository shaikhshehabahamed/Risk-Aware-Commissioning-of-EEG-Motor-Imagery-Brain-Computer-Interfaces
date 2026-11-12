from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import mne
import numpy as np

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PROJECT_SRC))

from rabp_bmi.channel_selection import select_channel_indices
from rabp_bmi.config import load_config
from rabp_bmi.dataset import discover_subjects
from rabp_bmi.evaluation import make_classifier
from rabp_bmi.features import extract_subject_logvar_features
from rabp_bmi.quality import DEFAULT_EXCLUDED_SUBJECTS


TARGET_COVERAGES = (1.0, 0.6, 0.4, 0.2)
DEFAULT_M_VALUES = (2, 4, 6, 8, 10, 12, 15)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def policy_budget_pairs() -> tuple[tuple[str, int], ...]:
    return (
        ("topk_effect", 8),
        ("topk_effect", 16),
        ("topk_effect", 21),
        ("topk_effect", 32),
        ("central_parietal_topk_effect", 21),
        ("central_parietal_topk_effect", 32),
        ("fixed_motor", 21),
        ("all", 64),
    )


def calibration_mask(
    runs: np.ndarray,
    y: np.ndarray,
    selected_runs: list[int],
    per_class_per_run: int,
) -> np.ndarray:
    mask = np.zeros(len(y), dtype=bool)
    for run in selected_runs:
        for label in (0, 1):
            idx = np.flatnonzero((runs == run) & (y == label))
            mask[idx[:per_class_per_run]] = True
    return mask


def threshold_for_coverage(confidence: np.ndarray, target_coverage: float) -> float:
    if target_coverage >= 1.0:
        return 0.0
    quantile = 1.0 - target_coverage
    return float(np.quantile(confidence, quantile, method="higher"))


def fit_predict_logistic(x_train, y_train, x_test) -> np.ndarray:
    clf = make_classifier("logistic")
    clf.fit(x_train, y_train)
    return clf.predict_proba(x_test)[:, 1]


def safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def summarize_trials(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int, int, float], list[dict]] = {}
    for row in rows:
        key = (
            str(row["policy"]),
            int(row["budget"]),
            int(row["m_per_class_per_run"]),
            float(row["target_coverage"]),
        )
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict] = []
    for (policy, budget, m, target), subset in sorted(groups.items()):
        accepted = [row for row in subset if row["accepted"] == 1]
        n_total = len(subset)
        n_accepted = len(accepted)

        tp = sum(1 for row in accepted if row["y_true"] == 1 and row["y_pred"] == 1)
        tn = sum(1 for row in accepted if row["y_true"] == 0 and row["y_pred"] == 0)
        fp = sum(1 for row in accepted if row["y_true"] == 0 and row["y_pred"] == 1)
        fn = sum(1 for row in accepted if row["y_true"] == 1 and row["y_pred"] == 0)

        total_rest = sum(1 for row in subset if row["y_true"] == 0)
        total_intent = sum(1 for row in subset if row["y_true"] == 1)
        accepted_rest = sum(1 for row in accepted if row["y_true"] == 0)
        accepted_intent = sum(1 for row in accepted if row["y_true"] == 1)

        sensitivity = safe_rate(tp, tp + fn)
        specificity = safe_rate(tn, tn + fp)
        if np.isnan(sensitivity) or np.isnan(specificity):
            accepted_ba = float("nan")
        else:
            accepted_ba = 0.5 * (sensitivity + specificity)

        actual_calibration_trials = int(
            round(
                np.mean(
                    [
                        row["actual_calibration_trials"]
                        for row in subset
                        if row["trial_in_fold"] == 0
                    ]
                )
            )
        )

        summary_rows.append(
            {
                "policy": policy,
                "budget": budget,
                "m_per_class_per_run": m,
                "nominal_calibration_trials": 4 * m,
                "mean_actual_calibration_trials": actual_calibration_trials,
                "target_coverage": target,
                "actual_coverage": safe_rate(n_accepted, n_total),
                "n_total": n_total,
                "n_accepted": n_accepted,
                "accepted_accuracy": safe_rate(tp + tn, n_accepted),
                "accepted_balanced_accuracy": accepted_ba,
                "accepted_sensitivity": sensitivity,
                "accepted_specificity": specificity,
                "false_command_rate_per_rest_opportunity": safe_rate(fp, total_rest),
                "missed_intent_rate_per_intent_opportunity": safe_rate(fn, total_intent),
                "accepted_rest_trials": accepted_rest,
                "accepted_intent_trials": accepted_intent,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            }
        )
    return summary_rows


def main() -> None:
    mne.set_log_level("ERROR")
    parser = argparse.ArgumentParser(
        description="Joint channel and calibration-trial budget frontier."
    )
    parser.add_argument("--config", default="configs/local.json")
    parser.add_argument("--m-values", default=",".join(str(v) for v in DEFAULT_M_VALUES))
    parser.add_argument("--tmin", type=float, default=1.5)
    parser.add_argument("--tmax", type=float, default=4.0)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--include-known-problematic", action="store_true")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    subjects = discover_subjects(cfg.dataset_root)
    if not args.include_known_problematic:
        subjects = [s for s in subjects if s not in DEFAULT_EXCLUDED_SUBJECTS]
    if args.max_subjects is not None:
        subjects = subjects[: args.max_subjects]

    m_values = parse_int_list(args.m_values)
    pairs = policy_budget_pairs()

    trial_rows: list[dict] = []
    selected_rows: list[dict] = []

    for subject in subjects:
        data = extract_subject_logvar_features(
            dataset_root=cfg.dataset_root,
            subject=subject,
            task_runs=cfg.task_runs,
            channel_set="all",
            label_mode="intent_rest",
            tmin=args.tmin,
            tmax=args.tmax,
        )
        unique_runs = sorted(np.unique(data.runs).tolist())

        for outer_test_run in unique_runs:
            outer_train_runs = [run for run in unique_runs if run != outer_test_run]
            test_mask = data.runs == outer_test_run

            for m in m_values:
                outer_calib_mask = calibration_mask(
                    data.runs,
                    data.y,
                    selected_runs=outer_train_runs,
                    per_class_per_run=m,
                )
                actual_calibration_trials = int(outer_calib_mask.sum())

                for policy, budget in pairs:
                    final_indices = select_channel_indices(
                        data.x[outer_calib_mask],
                        data.y[outer_calib_mask],
                        data.channel_names,
                        policy=policy,
                        budget=budget,
                    )
                    selected_names = [data.channel_names[idx] for idx in final_indices]
                    selected_rows.append(
                        {
                            "subject": subject,
                            "outer_test_run": int(outer_test_run),
                            "policy": policy,
                            "budget": budget,
                            "m_per_class_per_run": m,
                            "actual_calibration_trials": actual_calibration_trials,
                            "n_selected": len(final_indices),
                            "selected_channels": " ".join(selected_names),
                        }
                    )

                    val_confidences: list[float] = []
                    for val_run in outer_train_runs:
                        inner_train_runs = [run for run in outer_train_runs if run != val_run]
                        inner_train_mask = calibration_mask(
                            data.runs,
                            data.y,
                            selected_runs=inner_train_runs,
                            per_class_per_run=m,
                        )
                        val_mask = calibration_mask(
                            data.runs,
                            data.y,
                            selected_runs=[val_run],
                            per_class_per_run=m,
                        )
                        inner_indices = select_channel_indices(
                            data.x[inner_train_mask],
                            data.y[inner_train_mask],
                            data.channel_names,
                            policy=policy,
                            budget=budget,
                        )
                        val_proba = fit_predict_logistic(
                            data.x[inner_train_mask][:, inner_indices],
                            data.y[inner_train_mask],
                            data.x[val_mask][:, inner_indices],
                        )
                        val_confidences.extend(
                            np.maximum(val_proba, 1.0 - val_proba).tolist()
                        )

                    thresholds = {
                        target: threshold_for_coverage(np.array(val_confidences), target)
                        for target in TARGET_COVERAGES
                    }

                    test_proba = fit_predict_logistic(
                        data.x[outer_calib_mask][:, final_indices],
                        data.y[outer_calib_mask],
                        data.x[test_mask][:, final_indices],
                    )
                    y_true = data.y[test_mask]
                    confidence = np.maximum(test_proba, 1.0 - test_proba)
                    y_pred = (test_proba >= 0.5).astype(int)

                    for target, threshold in thresholds.items():
                        accepted = confidence >= threshold
                        for trial_idx, (yt, yp, pr, conf, acc) in enumerate(
                            zip(y_true, y_pred, test_proba, confidence, accepted, strict=True)
                        ):
                            trial_rows.append(
                                {
                                    "subject": subject,
                                    "test_run": int(outer_test_run),
                                    "trial_in_fold": trial_idx,
                                    "policy": policy,
                                    "budget": budget,
                                    "m_per_class_per_run": m,
                                    "nominal_calibration_trials": 4 * m,
                                    "actual_calibration_trials": actual_calibration_trials,
                                    "n_selected": len(final_indices),
                                    "target_coverage": target,
                                    "calibrated_threshold": threshold,
                                    "y_true": int(yt),
                                    "y_pred": int(yp),
                                    "proba_intent": float(pr),
                                    "confidence": float(conf),
                                    "accepted": int(acc),
                                }
                            )

    label = f"calibration_frontier_logvar_intent_rest_t{args.tmin:g}-{args.tmax:g}"
    if args.max_subjects is not None:
        label += f"_first{args.max_subjects}"
    out_dir = cfg.run_root / "calibration_frontier" / label
    trial_path = out_dir / "trial_results.csv"
    summary_path = out_dir / "summary.csv"
    selected_path = out_dir / "selected_channels.csv"

    summary_rows = summarize_trials(trial_rows)
    write_csv(trial_path, trial_rows)
    write_csv(summary_path, summary_rows)
    write_csv(selected_path, selected_rows)

    print(f"subjects={len(subjects)}")
    print(f"m_values={list(m_values)}")
    print(f"policy_budget_pairs={len(pairs)}")
    print(f"trial_rows={len(trial_rows)}")
    print(f"trial_results={trial_path}")
    print(f"summary={summary_path}")
    print(f"selected_channels={selected_path}")
    for row in summary_rows:
        if (
            row["policy"] in {"topk_effect", "central_parietal_topk_effect", "all"}
            and row["budget"] in {21, 32, 64}
            and row["m_per_class_per_run"] in {4, 8, 15}
            and row["target_coverage"] in {0.6, 0.4}
        ):
            print(
                "policy={policy} k={budget} m={m_per_class_per_run} "
                "target={target_coverage:.1f} actual={actual_coverage:.3f} "
                "ba={accepted_balanced_accuracy:.3f} "
                "fcr={false_command_rate_per_rest_opportunity:.3f}".format(**row)
            )


if __name__ == "__main__":
    main()

