from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def aggregate_selected(selected: pd.DataFrame, all_folds: pd.DataFrame) -> dict[str, float | int]:
    tp = int(selected["test_tp"].sum())
    tn = int(selected["test_tn"].sum())
    fp = int(selected["test_fp"].sum())
    fn = int(selected["test_fn"].sum())
    total_trials = int(selected["test_total_trials"].sum())
    total_rest = int(selected["test_total_rest"].sum())
    total_intent = int(selected["test_total_intent"].sum())
    n_accepted = int(selected["test_n_accepted"].sum())

    sensitivity = safe_rate(tp, tp + fn)
    specificity = safe_rate(tn, tn + fp)
    accepted_ba = (
        float("nan")
        if np.isnan(sensitivity) or np.isnan(specificity)
        else 0.5 * (sensitivity + specificity)
    )

    all_total_trials = int(all_folds[["subject", "outer_test_run"]].drop_duplicates().shape[0])
    # Each held-out run has variable but known trial count; use max per fold group once.
    all_fold_totals = all_folds.groupby(["subject", "outer_test_run"]).first()
    all_trial_count = int(all_fold_totals["test_total_trials"].sum())
    all_rest_count = int(all_fold_totals["test_total_rest"].sum())

    return {
        "eligible_folds": int(len(selected)),
        "all_outer_folds": all_total_trials,
        "deployment_rate": safe_rate(len(selected), all_total_trials),
        "deployed_total_trials": total_trials,
        "deployed_n_accepted": n_accepted,
        "deployed_actual_coverage": safe_rate(n_accepted, total_trials),
        "deployed_accepted_accuracy": safe_rate(tp + tn, n_accepted),
        "deployed_accepted_balanced_accuracy": accepted_ba,
        "deployed_accepted_sensitivity": sensitivity,
        "deployed_accepted_specificity": specificity,
        "deployed_false_command_rate_per_rest_opportunity": safe_rate(fp, total_rest),
        "deployed_wrong_rest_rate_per_intent_opportunity": safe_rate(fn, total_intent),
        "deployed_intent_command_acceptance_rate": safe_rate(tp, total_intent),
        "unconditional_actual_coverage": safe_rate(n_accepted, all_trial_count),
        "unconditional_false_command_rate": safe_rate(fp, all_rest_count),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select one readiness-feasible policy per held-out session."
    )
    parser.add_argument("fold_results_csv")
    parser.add_argument("--min-ba-values", default="0.60,0.65,0.70,0.75")
    parser.add_argument("--max-fcr-values", default="0.08,0.10,0.12,0.15")
    parser.add_argument("--min-coverage-values", default="0.15,0.20,0.25")
    parser.add_argument("--min-accepted", type=int, default=8)
    parser.add_argument("--channel-cost", type=float, default=1.0)
    parser.add_argument("--calibration-cost", type=float, default=0.25)
    parser.add_argument("--calib-fcr-penalty", type=float, default=80.0)
    parser.add_argument("--calib-ba-reward", type=float, default=20.0)
    parser.add_argument("--calib-coverage-reward", type=float, default=5.0)
    args = parser.parse_args()

    path = Path(args.fold_results_csv)
    df = pd.read_csv(path)
    min_ba_values = parse_float_list(args.min_ba_values)
    max_fcr_values = parse_float_list(args.max_fcr_values)
    min_cov_values = parse_float_list(args.min_coverage_values)

    rows: list[dict] = []
    selected_rows: list[dict] = []
    fold_keys = ["subject", "outer_test_run"]

    for min_ba in min_ba_values:
        for max_fcr in max_fcr_values:
            for min_cov in min_cov_values:
                feasible = df[
                    (df["calib_accepted_balanced_accuracy"] >= min_ba)
                    & (df["calib_false_command_rate_per_rest_opportunity"] <= max_fcr)
                    & (df["calib_actual_coverage"] >= min_cov)
                    & (df["calib_n_accepted"] >= args.min_accepted)
                ].copy()
                if feasible.empty:
                    continue

                feasible["decision_cost"] = (
                    args.channel_cost * feasible["budget"]
                    + args.calibration_cost * feasible["nominal_calibration_trials"]
                    + args.calib_fcr_penalty
                    * feasible["calib_false_command_rate_per_rest_opportunity"]
                    - args.calib_ba_reward * feasible["calib_accepted_balanced_accuracy"]
                    - args.calib_coverage_reward * feasible["calib_actual_coverage"]
                )
                selected = (
                    feasible.sort_values(
                        fold_keys + ["decision_cost", "budget", "nominal_calibration_trials"]
                    )
                    .groupby(fold_keys, as_index=False)
                    .first()
                )
                metrics = aggregate_selected(selected, df)
                row = {
                    "gate_min_calib_ba": min_ba,
                    "gate_max_calib_fcr": max_fcr,
                    "gate_min_calib_coverage": min_cov,
                    "gate_min_accepted": args.min_accepted,
                    "mean_selected_budget": float(selected["budget"].mean()),
                    "median_selected_budget": float(selected["budget"].median()),
                    "mean_selected_calibration_trials": float(
                        selected["nominal_calibration_trials"].mean()
                    ),
                    "median_selected_calibration_trials": float(
                        selected["nominal_calibration_trials"].median()
                    ),
                    "mean_selected_calib_ba": float(
                        selected["calib_accepted_balanced_accuracy"].mean()
                    ),
                    "mean_selected_calib_fcr": float(
                        selected["calib_false_command_rate_per_rest_opportunity"].mean()
                    ),
                    "mean_selected_calib_coverage": float(
                        selected["calib_actual_coverage"].mean()
                    ),
                }
                row.update(metrics)
                rows.append(row)

                for _, selected_row in selected.iterrows():
                    selected_rows.append(
                        {
                            "gate_min_calib_ba": min_ba,
                            "gate_max_calib_fcr": max_fcr,
                            "gate_min_calib_coverage": min_cov,
                            "subject": selected_row["subject"],
                            "outer_test_run": int(selected_row["outer_test_run"]),
                            "candidate": selected_row["candidate"],
                            "policy": selected_row["policy"],
                            "budget": int(selected_row["budget"]),
                            "m_per_class_per_run": int(selected_row["m_per_class_per_run"]),
                            "nominal_calibration_trials": int(
                                selected_row["nominal_calibration_trials"]
                            ),
                            "decision_cost": float(selected_row["decision_cost"]),
                        }
                    )

    out_dir = path.parent
    summary_path = out_dir / "selected_policy_summary.csv"
    selected_path = out_dir / "selected_policy_assignments.csv"
    write_csv(summary_path, rows)
    write_csv(selected_path, selected_rows)

    summary = pd.DataFrame(rows)
    summary["score"] = (
        summary["deployed_accepted_balanced_accuracy"]
        - summary["deployed_false_command_rate_per_rest_opportunity"]
        + 0.10 * summary["deployed_actual_coverage"]
        + 0.25 * summary["deployment_rate"]
        - 0.001 * summary["mean_selected_budget"]
        - 0.0005 * summary["mean_selected_calibration_trials"]
    )
    summary = summary.sort_values(
        ["score", "deployed_accepted_balanced_accuracy", "deployment_rate"],
        ascending=False,
    )
    ranked_path = out_dir / "selected_policy_ranked.csv"
    summary.to_csv(ranked_path, index=False)

    print(f"summary={summary_path}")
    print(f"selected_assignments={selected_path}")
    print(f"ranked={ranked_path}")
    print(
        summary[
            [
                "gate_min_calib_ba",
                "gate_max_calib_fcr",
                "gate_min_calib_coverage",
                "deployment_rate",
                "mean_selected_budget",
                "mean_selected_calibration_trials",
                "deployed_actual_coverage",
                "deployed_accepted_balanced_accuracy",
                "deployed_false_command_rate_per_rest_opportunity",
                "unconditional_actual_coverage",
                "unconditional_false_command_rate",
                "score",
            ]
        ]
        .head(20)
        .to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )


if __name__ == "__main__":
    main()

