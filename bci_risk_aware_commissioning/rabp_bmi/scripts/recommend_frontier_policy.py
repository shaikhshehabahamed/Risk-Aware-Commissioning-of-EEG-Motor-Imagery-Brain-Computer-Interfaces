from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recommend a feasible channel/calibration/rejection policy."
    )
    parser.add_argument("summary_csv")
    parser.add_argument("--min-ba", type=float, default=0.75)
    parser.add_argument("--max-fcr", type=float, default=0.06)
    parser.add_argument("--min-coverage", type=float, default=0.20)
    parser.add_argument("--max-channels", type=int, default=64)
    parser.add_argument("--max-calibration-trials", type=int, default=10_000)
    parser.add_argument("--channel-cost", type=float, default=1.0)
    parser.add_argument("--calibration-cost", type=float, default=0.5)
    parser.add_argument("--fcr-penalty", type=float, default=100.0)
    parser.add_argument("--coverage-reward", type=float, default=10.0)
    parser.add_argument("--ba-reward", type=float, default=10.0)
    args = parser.parse_args()

    path = Path(args.summary_csv)
    df = pd.read_csv(path)
    df = df[
        (df["accepted_balanced_accuracy"] >= args.min_ba)
        & (df["false_command_rate_per_rest_opportunity"] <= args.max_fcr)
        & (df["actual_coverage"] >= args.min_coverage)
        & (df["budget"] <= args.max_channels)
        & (df["nominal_calibration_trials"] <= args.max_calibration_trials)
    ].copy()

    if df.empty:
        print("No feasible policy under the supplied constraints.")
        return

    df["objective_cost"] = (
        args.channel_cost * df["budget"]
        + args.calibration_cost * df["nominal_calibration_trials"]
        + args.fcr_penalty * df["false_command_rate_per_rest_opportunity"]
        - args.coverage_reward * df["actual_coverage"]
        - args.ba_reward * df["accepted_balanced_accuracy"]
    )
    df = df.sort_values(
        [
            "objective_cost",
            "nominal_calibration_trials",
            "budget",
            "false_command_rate_per_rest_opportunity",
        ],
        ascending=[True, True, True, True],
    )

    cols = [
        "policy",
        "budget",
        "m_per_class_per_run",
        "nominal_calibration_trials",
        "target_coverage",
        "actual_coverage",
        "accepted_balanced_accuracy",
        "false_command_rate_per_rest_opportunity",
        "missed_intent_rate_per_intent_opportunity",
        "objective_cost",
    ]
    print("Recommended policy:")
    print(df[cols].head(1).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nTop feasible alternatives:")
    print(df[cols].head(10).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()

