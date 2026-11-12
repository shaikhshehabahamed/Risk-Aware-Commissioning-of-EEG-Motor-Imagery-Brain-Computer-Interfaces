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


def safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def aggregate_rows(group: pd.DataFrame, all_outer_folds: int, all_trial_count: int, all_rest_count: int) -> dict[str, float | int]:
    tp = int(group["test_tp"].sum())
    tn = int(group["test_tn"].sum())
    fp = int(group["test_fp"].sum())
    fn = int(group["test_fn"].sum())
    total_trials = int(group["test_total_trials"].sum())
    total_rest = int(group["test_total_rest"].sum())
    total_intent = int(group["test_total_intent"].sum())
    n_accepted = int(group["test_n_accepted"].sum())

    sensitivity = safe_rate(tp, tp + fn)
    specificity = safe_rate(tn, tn + fp)
    accepted_ba = (
        float("nan")
        if np.isnan(sensitivity) or np.isnan(specificity)
        else 0.5 * (sensitivity + specificity)
    )

    return {
        "deployed_folds": int(len(group)),
        "all_outer_folds": all_outer_folds,
        "deployment_rate": safe_rate(len(group), all_outer_folds),
        "deployed_total_trials": total_trials,
        "deployed_n_accepted": n_accepted,
        "deployed_actual_coverage": safe_rate(n_accepted, total_trials),
        "deployed_accepted_accuracy": safe_rate(tp + tn, n_accepted),
        "deployed_accepted_balanced_accuracy": accepted_ba,
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
        description="Summarize meta-readiness gates and adaptive meta selectors."
    )
    parser.add_argument("meta_predictions_csv")
    parser.add_argument("--thresholds", default="0.2,0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--min-selected-prob", type=float, default=None)
    parser.add_argument("--channel-cost", type=float, default=1.0)
    parser.add_argument("--calibration-cost", type=float, default=0.25)
    parser.add_argument("--probability-reward", type=float, default=100.0)
    args = parser.parse_args()

    path = Path(args.meta_predictions_csv)
    df = pd.read_csv(path)
    thresholds = tuple(float(x.strip()) for x in args.thresholds.split(",") if x.strip())
    fold_first = df.groupby(["subject", "outer_test_run"]).first()
    all_outer_folds = len(fold_first)
    all_trial_count = int(fold_first["test_total_trials"].sum())
    all_rest_count = int(fold_first["test_total_rest"].sum())

    gate_rows: list[dict] = []
    for candidate, candidate_group in df.groupby("candidate"):
        for threshold in thresholds:
            deployed = candidate_group[
                candidate_group["meta_success_probability"] >= threshold
            ].copy()
            if deployed.empty:
                continue
            row = {
                "mode": "fixed_candidate_meta_gate",
                "candidate": candidate,
                "threshold": threshold,
                "mean_meta_probability": float(deployed["meta_success_probability"].mean()),
                "success_label_rate": float(deployed["success_label"].mean()),
            }
            row.update(aggregate_rows(deployed, all_outer_folds, all_trial_count, all_rest_count))
            gate_rows.append(row)

    selector_rows: list[dict] = []
    selected_assignments: list[dict] = []
    for threshold in thresholds:
        feasible = df[df["meta_success_probability"] >= threshold].copy()
        if feasible.empty:
            continue
        feasible["decision_cost"] = (
            args.channel_cost * feasible["budget"]
            + args.calibration_cost * feasible["nominal_calibration_trials"]
            - args.probability_reward * feasible["meta_success_probability"]
        )
        selected = (
            feasible.sort_values(
                ["subject", "outer_test_run", "decision_cost", "budget"]
            )
            .groupby(["subject", "outer_test_run"], as_index=False)
            .first()
        )
        row = {
            "mode": "adaptive_meta_selector",
            "candidate": "adaptive",
            "threshold": threshold,
            "mean_selected_budget": float(selected["budget"].mean()),
            "mean_selected_calibration_trials": float(
                selected["nominal_calibration_trials"].mean()
            ),
            "mean_meta_probability": float(selected["meta_success_probability"].mean()),
            "success_label_rate": float(selected["success_label"].mean()),
        }
        row.update(aggregate_rows(selected, all_outer_folds, all_trial_count, all_rest_count))
        selector_rows.append(row)
        for _, selected_row in selected.iterrows():
            selected_assignments.append(
                {
                    "threshold": threshold,
                    "subject": selected_row["subject"],
                    "outer_test_run": int(selected_row["outer_test_run"]),
                    "candidate": selected_row["candidate"],
                    "policy": selected_row["policy"],
                    "budget": int(selected_row["budget"]),
                    "m_per_class_per_run": int(selected_row["m_per_class_per_run"]),
                    "nominal_calibration_trials": int(
                        selected_row["nominal_calibration_trials"]
                    ),
                    "meta_success_probability": float(
                        selected_row["meta_success_probability"]
                    ),
                    "decision_cost": float(selected_row["decision_cost"]),
                    "success_label": int(selected_row["success_label"]),
                }
            )

    out_dir = path.parent
    gate_path = out_dir / "meta_gate_summary.csv"
    selector_path = out_dir / "meta_selector_summary.csv"
    assignments_path = out_dir / "meta_selector_assignments.csv"
    write_csv(gate_path, gate_rows)
    write_csv(selector_path, selector_rows)
    write_csv(assignments_path, selected_assignments)

    combined = pd.DataFrame(gate_rows + selector_rows)
    combined["score"] = (
        combined["deployed_accepted_balanced_accuracy"]
        - combined["deployed_false_command_rate_per_rest_opportunity"]
        + 0.10 * combined["deployed_actual_coverage"]
        + 0.25 * combined["deployment_rate"]
    )
    ranked = combined.sort_values(
        ["score", "deployed_accepted_balanced_accuracy", "deployment_rate"],
        ascending=False,
    )
    ranked_path = out_dir / "meta_readiness_ranked.csv"
    ranked.to_csv(ranked_path, index=False)

    print(f"gate_summary={gate_path}")
    print(f"selector_summary={selector_path}")
    print(f"selector_assignments={assignments_path}")
    print(f"ranked={ranked_path}")
    print(
        ranked[
            [
                "mode",
                "candidate",
                "threshold",
                "deployment_rate",
                "deployed_actual_coverage",
                "deployed_accepted_balanced_accuracy",
                "deployed_false_command_rate_per_rest_opportunity",
                "success_label_rate",
                "mean_meta_probability",
                "score",
            ]
        ]
        .head(25)
        .to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )


if __name__ == "__main__":
    main()

