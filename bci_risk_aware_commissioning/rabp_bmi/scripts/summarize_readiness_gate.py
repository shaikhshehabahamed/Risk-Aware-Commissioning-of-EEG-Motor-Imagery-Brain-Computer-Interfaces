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


def aggregate_metrics(group: pd.DataFrame, prefix: str) -> dict[str, float | int]:
    tp = int(group[f"{prefix}_tp"].sum())
    tn = int(group[f"{prefix}_tn"].sum())
    fp = int(group[f"{prefix}_fp"].sum())
    fn = int(group[f"{prefix}_fn"].sum())
    total_trials = int(group[f"{prefix}_total_trials"].sum())
    total_rest = int(group[f"{prefix}_total_rest"].sum())
    total_intent = int(group[f"{prefix}_total_intent"].sum())
    n_accepted = int(group[f"{prefix}_n_accepted"].sum())

    sensitivity = safe_rate(tp, tp + fn)
    specificity = safe_rate(tn, tn + fp)
    accepted_ba = (
        float("nan")
        if np.isnan(sensitivity) or np.isnan(specificity)
        else 0.5 * (sensitivity + specificity)
    )

    return {
        f"{prefix}_total_trials": total_trials,
        f"{prefix}_total_rest": total_rest,
        f"{prefix}_total_intent": total_intent,
        f"{prefix}_n_accepted": n_accepted,
        f"{prefix}_actual_coverage": safe_rate(n_accepted, total_trials),
        f"{prefix}_accepted_accuracy": safe_rate(tp + tn, n_accepted),
        f"{prefix}_accepted_balanced_accuracy": accepted_ba,
        f"{prefix}_accepted_sensitivity": sensitivity,
        f"{prefix}_accepted_specificity": specificity,
        f"{prefix}_false_command_rate_per_rest_opportunity": safe_rate(fp, total_rest),
        f"{prefix}_wrong_rest_rate_per_intent_opportunity": safe_rate(fn, total_intent),
        f"{prefix}_intent_command_acceptance_rate": safe_rate(tp, total_intent),
        f"{prefix}_tp": tp,
        f"{prefix}_tn": tn,
        f"{prefix}_fp": fp,
        f"{prefix}_fn": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply calibration-only readiness gates.")
    parser.add_argument("fold_results_csv")
    parser.add_argument("--min-ba-values", default="0.60,0.65,0.70,0.75")
    parser.add_argument("--max-fcr-values", default="0.08,0.10,0.12,0.15")
    parser.add_argument("--min-coverage-values", default="0.15,0.20,0.25")
    parser.add_argument("--min-accepted", type=int, default=8)
    parser.add_argument("--rank-min-deployment-rate", type=float, default=0.25)
    args = parser.parse_args()

    path = Path(args.fold_results_csv)
    df = pd.read_csv(path)
    min_ba_values = parse_float_list(args.min_ba_values)
    max_fcr_values = parse_float_list(args.max_fcr_values)
    min_coverage_values = parse_float_list(args.min_coverage_values)

    rows: list[dict] = []
    for candidate, candidate_group in df.groupby("candidate"):
        all_metrics = aggregate_metrics(candidate_group, "test")
        n_folds = len(candidate_group)
        for min_ba in min_ba_values:
            for max_fcr in max_fcr_values:
                for min_cov in min_coverage_values:
                    deploy_mask = (
                        (candidate_group["calib_accepted_balanced_accuracy"] >= min_ba)
                        & (
                            candidate_group["calib_false_command_rate_per_rest_opportunity"]
                            <= max_fcr
                        )
                        & (candidate_group["calib_actual_coverage"] >= min_cov)
                        & (candidate_group["calib_n_accepted"] >= args.min_accepted)
                    )
                    deployed = candidate_group[deploy_mask]
                    if deployed.empty:
                        continue

                    deployed_metrics = aggregate_metrics(deployed, "test")
                    row = {
                        "candidate": candidate,
                        "policy": str(candidate_group["policy"].iloc[0]),
                        "budget": int(candidate_group["budget"].iloc[0]),
                        "m_per_class_per_run": int(candidate_group["m_per_class_per_run"].iloc[0]),
                        "nominal_calibration_trials": int(
                            candidate_group["nominal_calibration_trials"].iloc[0]
                        ),
                        "target_coverage": float(candidate_group["target_coverage"].iloc[0]),
                        "gate_min_calib_ba": min_ba,
                        "gate_max_calib_fcr": max_fcr,
                        "gate_min_calib_coverage": min_cov,
                        "gate_min_accepted": args.min_accepted,
                        "n_folds": n_folds,
                        "deployed_folds": len(deployed),
                        "deployment_rate": safe_rate(len(deployed), n_folds),
                        "mean_deployed_calib_ba": float(
                            deployed["calib_accepted_balanced_accuracy"].mean()
                        ),
                        "mean_deployed_calib_fcr": float(
                            deployed[
                                "calib_false_command_rate_per_rest_opportunity"
                            ].mean()
                        ),
                        "mean_deployed_calib_coverage": float(
                            deployed["calib_actual_coverage"].mean()
                        ),
                        "unconditional_test_actual_coverage": safe_rate(
                            deployed_metrics["test_n_accepted"],
                            all_metrics["test_total_trials"],
                        ),
                        "unconditional_test_fcr": safe_rate(
                            deployed_metrics["test_fp"],
                            all_metrics["test_total_rest"],
                        ),
                    }
                    row.update({f"deployed_{key}": value for key, value in deployed_metrics.items()})
                    row.update({f"all_sessions_{key}": value for key, value in all_metrics.items()})
                    row["fcr_reduction_vs_all_sessions"] = (
                        all_metrics["test_false_command_rate_per_rest_opportunity"]
                        - deployed_metrics["test_false_command_rate_per_rest_opportunity"]
                    )
                    row["ba_gain_vs_all_sessions"] = (
                        deployed_metrics["test_accepted_balanced_accuracy"]
                        - all_metrics["test_accepted_balanced_accuracy"]
                    )
                    rows.append(row)

    out_dir = path.parent
    out_path = out_dir / "readiness_gate_summary.csv"
    write_csv(out_path, rows)

    summary = pd.DataFrame(rows)
    summary["score"] = (
        summary["deployed_test_accepted_balanced_accuracy"]
        - summary["deployed_test_false_command_rate_per_rest_opportunity"]
        + 0.10 * summary["deployed_test_actual_coverage"]
        + 0.25 * summary["deployment_rate"]
        - 0.001 * summary["budget"]
        - 0.0005 * summary["nominal_calibration_trials"]
    )
    summary = summary.sort_values(
        [
            "score",
            "deployed_test_accepted_balanced_accuracy",
            "deployment_rate",
        ],
        ascending=False,
    )
    ranked_path = out_dir / "readiness_gate_ranked.csv"
    summary.to_csv(ranked_path, index=False)

    print(f"summary={out_path}")
    print(f"ranked={ranked_path}")
    display = summary[summary["deployment_rate"] >= args.rank_min_deployment_rate]
    if display.empty:
        display = summary
    print(
        display[
            [
                "candidate",
                "gate_min_calib_ba",
                "gate_max_calib_fcr",
                "gate_min_calib_coverage",
                "deployment_rate",
                "deployed_test_actual_coverage",
                "deployed_test_accepted_balanced_accuracy",
                "deployed_test_false_command_rate_per_rest_opportunity",
                "unconditional_test_actual_coverage",
                "unconditional_test_fcr",
                "score",
            ]
        ]
        .head(20)
        .to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )


if __name__ == "__main__":
    main()
