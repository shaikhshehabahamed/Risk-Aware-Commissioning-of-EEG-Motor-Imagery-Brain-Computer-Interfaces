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


DEFAULT_CANDIDATES = (
    ("all", 64, 15, 0.4),
    ("topk_effect", 8, 12, 0.4),
    ("topk_effect", 16, 15, 0.4),
    ("topk_effect", 21, 12, 0.4),
    ("topk_effect", 21, 15, 0.4),
    ("topk_effect", 32, 8, 0.4),
    ("central_parietal_topk_effect", 21, 15, 0.4),
)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def fit_predict_classifier(
    classifier: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> np.ndarray:
    clf = make_classifier(classifier)
    clf.fit(x_train, y_train)
    return clf.predict_proba(x_test)[:, 1]


def safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def accepted_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    confidence = np.maximum(proba, 1.0 - proba)
    y_pred = (proba >= 0.5).astype(int)
    accepted = confidence >= threshold

    accepted_y = y_true[accepted]
    accepted_pred = y_pred[accepted]

    tp = int(((accepted_y == 1) & (accepted_pred == 1)).sum())
    tn = int(((accepted_y == 0) & (accepted_pred == 0)).sum())
    fp = int(((accepted_y == 0) & (accepted_pred == 1)).sum())
    fn = int(((accepted_y == 1) & (accepted_pred == 0)).sum())

    total_trials = int(len(y_true))
    total_rest = int((y_true == 0).sum())
    total_intent = int((y_true == 1).sum())
    n_accepted = int(accepted.sum())
    accepted_rest = int((accepted_y == 0).sum())
    accepted_intent = int((accepted_y == 1).sum())

    sensitivity = safe_rate(tp, tp + fn)
    specificity = safe_rate(tn, tn + fp)
    accepted_ba = (
        float("nan")
        if np.isnan(sensitivity) or np.isnan(specificity)
        else 0.5 * (sensitivity + specificity)
    )

    return {
        "total_trials": total_trials,
        "total_rest": total_rest,
        "total_intent": total_intent,
        "n_accepted": n_accepted,
        "accepted_rest": accepted_rest,
        "accepted_intent": accepted_intent,
        "actual_coverage": safe_rate(n_accepted, total_trials),
        "accepted_accuracy": safe_rate(tp + tn, n_accepted),
        "accepted_balanced_accuracy": accepted_ba,
        "accepted_sensitivity": sensitivity,
        "accepted_specificity": specificity,
        "false_command_rate_per_rest_opportunity": safe_rate(fp, total_rest),
        "wrong_rest_rate_per_intent_opportunity": safe_rate(fn, total_intent),
        "intent_command_acceptance_rate": safe_rate(tp, total_intent),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def candidate_id(policy: str, budget: int, m: int, target: float) -> str:
    return f"{policy}_k{budget}_m{m}_target{target:g}"


def main() -> None:
    mne.set_log_level("ERROR")
    parser = argparse.ArgumentParser(
        description="Estimate calibration-only readiness and evaluate held-out runs."
    )
    parser.add_argument("--config", default="configs/local.json")
    parser.add_argument("--tmin", type=float, default=1.5)
    parser.add_argument("--tmax", type=float, default=4.0)
    parser.add_argument("--classifier", choices=("logistic", "lda"), default="logistic")
    parser.add_argument("--target-coverages", default="0.4")
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--include-known-problematic", action="store_true")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    target_coverages = parse_float_list(args.target_coverages)
    subjects = discover_subjects(cfg.dataset_root)
    if not args.include_known_problematic:
        subjects = [s for s in subjects if s not in DEFAULT_EXCLUDED_SUBJECTS]
    if args.max_subjects is not None:
        subjects = subjects[: args.max_subjects]

    fold_rows: list[dict] = []
    selected_rows: list[dict] = []
    seen_candidates: set[tuple[str, int, int]] = set()
    base_candidates: list[tuple[str, int, int]] = []
    for policy, budget, m, _ in DEFAULT_CANDIDATES:
        key = (policy, budget, m)
        if key not in seen_candidates:
            seen_candidates.add(key)
            base_candidates.append(key)

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

            for policy, budget, m in base_candidates:
                outer_calib_mask = calibration_mask(
                    data.runs,
                    data.y,
                    selected_runs=outer_train_runs,
                    per_class_per_run=m,
                )
                final_indices = select_channel_indices(
                    data.x[outer_calib_mask],
                    data.y[outer_calib_mask],
                    data.channel_names,
                    policy=policy,
                    budget=budget,
                )
                val_y: list[int] = []
                val_proba: list[float] = []
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
                    proba = fit_predict_classifier(
                        args.classifier,
                        data.x[inner_train_mask][:, inner_indices],
                        data.y[inner_train_mask],
                        data.x[val_mask][:, inner_indices],
                    )
                    val_y.extend(data.y[val_mask].astype(int).tolist())
                    val_proba.extend(proba.tolist())

                val_y_array = np.array(val_y, dtype=int)
                val_proba_array = np.array(val_proba, dtype=float)
                test_proba = fit_predict_classifier(
                    args.classifier,
                    data.x[outer_calib_mask][:, final_indices],
                    data.y[outer_calib_mask],
                    data.x[test_mask][:, final_indices],
                )

                for target in target_coverages:
                    selected_rows.append(
                        {
                            "candidate": candidate_id(policy, budget, m, target),
                            "subject": subject,
                            "outer_test_run": int(outer_test_run),
                            "policy": policy,
                            "budget": budget,
                            "m_per_class_per_run": m,
                            "nominal_calibration_trials": 4 * m,
                            "actual_calibration_trials": int(outer_calib_mask.sum()),
                            "target_coverage": target,
                            "n_selected": len(final_indices),
                            "selected_channels": " ".join(
                                data.channel_names[idx] for idx in final_indices
                            ),
                        }
                    )
                    threshold = threshold_for_coverage(
                        np.maximum(val_proba_array, 1.0 - val_proba_array),
                        target,
                    )
                    calib_metrics = accepted_metrics(
                        val_y_array, val_proba_array, threshold
                    )
                    test_metrics = accepted_metrics(
                        data.y[test_mask], test_proba, threshold
                    )

                    row = {
                        "candidate": candidate_id(policy, budget, m, target),
                        "subject": subject,
                        "outer_test_run": int(outer_test_run),
                        "policy": policy,
                        "budget": budget,
                        "m_per_class_per_run": m,
                        "nominal_calibration_trials": 4 * m,
                        "actual_calibration_trials": int(outer_calib_mask.sum()),
                        "target_coverage": target,
                        "calibrated_threshold": threshold,
                        "n_selected": len(final_indices),
                    }
                    row.update({f"calib_{key}": value for key, value in calib_metrics.items()})
                    row.update({f"test_{key}": value for key, value in test_metrics.items()})
                    fold_rows.append(row)

    label = f"readiness_gate_logvar_intent_rest_t{args.tmin:g}-{args.tmax:g}"
    if args.classifier != "logistic":
        label += f"_{args.classifier}"
    if args.output_suffix:
        label += f"_{args.output_suffix}"
    if args.max_subjects is not None:
        label += f"_first{args.max_subjects}"
    out_dir = cfg.run_root / "readiness_gate" / label
    fold_path = out_dir / "fold_results.csv"
    selected_path = out_dir / "selected_channels.csv"
    write_csv(fold_path, fold_rows)
    write_csv(selected_path, selected_rows)

    print(f"subjects={len(subjects)}")
    print(f"base_candidates={len(base_candidates)}")
    print(f"target_coverages={list(target_coverages)}")
    print(f"fold_rows={len(fold_rows)}")
    print(f"fold_results={fold_path}")
    print(f"selected_channels={selected_path}")


if __name__ == "__main__":
    main()
