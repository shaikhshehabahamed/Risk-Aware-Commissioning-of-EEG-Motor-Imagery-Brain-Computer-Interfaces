from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def pareto_mask(df: pd.DataFrame) -> pd.Series:
    dominated = []
    for idx, row in df.iterrows():
        others = df.drop(index=idx)
        is_dominated = (
            (others["budget"] <= row["budget"])
            & (others["nominal_calibration_trials"] <= row["nominal_calibration_trials"])
            & (others["false_command_rate_per_rest_opportunity"] <= row["false_command_rate_per_rest_opportunity"])
            & (others["accepted_balanced_accuracy"] >= row["accepted_balanced_accuracy"])
            & (others["actual_coverage"] >= row["actual_coverage"])
            & (
                (others["budget"] < row["budget"])
                | (others["nominal_calibration_trials"] < row["nominal_calibration_trials"])
                | (
                    others["false_command_rate_per_rest_opportunity"]
                    < row["false_command_rate_per_rest_opportunity"]
                )
                | (others["accepted_balanced_accuracy"] > row["accepted_balanced_accuracy"])
                | (others["actual_coverage"] > row["actual_coverage"])
            )
        ).any()
        dominated.append(is_dominated)
    return ~pd.Series(dominated, index=df.index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize calibration frontier.")
    parser.add_argument("run_dir")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    df = pd.read_csv(run_dir / "summary.csv")
    df["channel_reduction_vs_64"] = 1.0 - df["budget"] / 64.0
    df["safety_score"] = (
        df["accepted_balanced_accuracy"]
        - df["false_command_rate_per_rest_opportunity"]
        + 0.10 * df["actual_coverage"]
        - 0.001 * df["budget"]
        - 0.0005 * df["nominal_calibration_trials"]
    )

    compact = df[
        [
            "policy",
            "budget",
            "m_per_class_per_run",
            "nominal_calibration_trials",
            "target_coverage",
            "actual_coverage",
            "accepted_balanced_accuracy",
            "false_command_rate_per_rest_opportunity",
            "missed_intent_rate_per_intent_opportunity",
            "channel_reduction_vs_64",
            "safety_score",
        ]
    ].copy()
    compact = compact.sort_values(
        [
            "target_coverage",
            "safety_score",
            "accepted_balanced_accuracy",
            "actual_coverage",
        ],
        ascending=[False, False, False, False],
    )
    compact_path = run_dir / "frontier_ranked_summary.csv"
    compact.to_csv(compact_path, index=False)

    feasible = compact[
        (compact["accepted_balanced_accuracy"] >= 0.75)
        & (compact["false_command_rate_per_rest_opportunity"] <= 0.06)
        & (compact["actual_coverage"] >= 0.20)
    ].copy()
    feasible = feasible.sort_values(
        [
            "nominal_calibration_trials",
            "budget",
            "false_command_rate_per_rest_opportunity",
            "accepted_balanced_accuracy",
        ],
        ascending=[True, True, True, False],
    )
    feasible_path = run_dir / "feasible_ba75_fcr06_cov20.csv"
    feasible.to_csv(feasible_path, index=False)

    pareto = compact[pareto_mask(compact)].copy()
    pareto = pareto.sort_values(
        [
            "target_coverage",
            "nominal_calibration_trials",
            "budget",
            "false_command_rate_per_rest_opportunity",
        ],
        ascending=[False, True, True, True],
    )
    pareto_path = run_dir / "pareto_candidates.csv"
    pareto.to_csv(pareto_path, index=False)

    print(f"ranked_summary={compact_path}")
    print(f"feasible_table={feasible_path}")
    print(f"pareto_candidates={pareto_path}")

    print("\nFeasible: accepted BA >= 0.75, FCR <= 0.06, actual coverage >= 0.20")
    if feasible.empty:
        print("none")
    else:
        print(
            feasible[
                [
                    "policy",
                    "budget",
                    "m_per_class_per_run",
                    "nominal_calibration_trials",
                    "target_coverage",
                    "actual_coverage",
                    "accepted_balanced_accuracy",
                    "false_command_rate_per_rest_opportunity",
                ]
            ].head(20).to_string(index=False, float_format=lambda x: f"{x:.3f}")
        )

    print("\nBest rows by target coverage")
    for target in sorted(compact["target_coverage"].unique(), reverse=True):
        part = compact[compact["target_coverage"] == target]
        print(f"\ntarget={target:.1f}")
        print(
            part[
                [
                    "policy",
                    "budget",
                    "m_per_class_per_run",
                    "nominal_calibration_trials",
                    "actual_coverage",
                    "accepted_balanced_accuracy",
                    "false_command_rate_per_rest_opportunity",
                    "safety_score",
                ]
            ].head(8).to_string(index=False, float_format=lambda x: f"{x:.3f}")
        )


if __name__ == "__main__":
    main()

