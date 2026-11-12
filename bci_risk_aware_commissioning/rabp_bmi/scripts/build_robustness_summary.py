from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PROJECT_SRC))

from rabp_bmi.config import load_config


WINDOW_POLICIES = [
    ("all", 64),
    ("topk_effect", 16),
    ("topk_effect", 21),
    ("topk_effect", 32),
    ("fixed_motor", 21),
    ("motor_topk_effect", 16),
    ("motor_topk_effect", 21),
]

VALIDATION_MODES = [
    "all64_m15_no_gate",
    "topk16_m15_no_gate",
    "topk21_m15_no_gate",
    "central21_m15_no_gate",
    "fixed_motor21_m15_no_gate",
    "topk21_m15_rule_t03_ba065_fcr010",
    "topk16_m15_rule_t03_ba070_fcr010",
    "topk16_m15_rule_ba075_fcr010",
    "topk21_m15_t03_meta_p060",
]

MODE_TO_REGION_KEY = {
    "all64_m15_no_gate": ("calibration_limited", "all", 64, 15),
    "topk16_m15_no_gate": ("calibration_limited", "topk_effect", 16, 15),
    "topk21_m15_no_gate": ("calibration_limited", "topk_effect", 21, 15),
    "central21_m15_no_gate": (
        "calibration_limited",
        "central_parietal_topk_effect",
        21,
        15,
    ),
    "fixed_motor21_m15_no_gate": ("calibration_limited", "fixed_motor", 21, 15),
    "topk21_m15_rule_t03_ba065_fcr010": (
        "calibration_limited",
        "topk_effect",
        21,
        15,
    ),
    "topk16_m15_rule_t03_ba070_fcr010": (
        "calibration_limited",
        "topk_effect",
        16,
        15,
    ),
    "topk16_m15_rule_ba075_fcr010": (
        "calibration_limited",
        "topk_effect",
        16,
        15,
    ),
    "topk21_m15_t03_meta_p060": ("calibration_limited", "topk_effect", 21, 15),
}


def report_date() -> str:
    return datetime.now().date().isoformat()


def row_for_policy(summary: pd.DataFrame, policy: str, budget: int, target: float) -> pd.Series:
    rows = summary[
        (summary["policy"] == policy)
        & (summary["budget"] == budget)
        & (summary["target_coverage"] == target)
    ]
    if rows.empty:
        raise ValueError(f"Missing summary row: {policy} {budget} target {target}")
    return rows.iloc[0]


def build_window_sensitivity(run_root: Path) -> pd.DataFrame:
    original = pd.read_csv(
        run_root
        / "budgeted_rejection"
        / "budgeted_logvar_intent_rest_policies_v2_t0.5-3.5"
        / "summary.csv"
    )
    delayed = pd.read_csv(
        run_root
        / "budgeted_rejection"
        / "budgeted_logvar_intent_rest_policies_v2_t1.5-4"
        / "summary.csv"
    )
    rows: list[dict[str, object]] = []
    for policy, budget in WINDOW_POLICIES:
        original_row = row_for_policy(original, policy, budget, 0.4)
        delayed_row = row_for_policy(delayed, policy, budget, 0.4)
        rows.append(
            {
                "policy": policy,
                "budget": budget,
                "target_coverage": 0.4,
                "original_window": "0.5-3.5",
                "delayed_window": "1.5-4.0",
                "original_actual_coverage": original_row["actual_coverage"],
                "delayed_actual_coverage": delayed_row["actual_coverage"],
                "coverage_change_delayed_minus_original": delayed_row[
                    "actual_coverage"
                ]
                - original_row["actual_coverage"],
                "original_accepted_ba": original_row["accepted_balanced_accuracy"],
                "delayed_accepted_ba": delayed_row["accepted_balanced_accuracy"],
                "ba_change_delayed_minus_original": delayed_row[
                    "accepted_balanced_accuracy"
                ]
                - original_row["accepted_balanced_accuracy"],
                "original_fcr": original_row[
                    "false_command_rate_per_rest_opportunity"
                ],
                "delayed_fcr": delayed_row["false_command_rate_per_rest_opportunity"],
                "fcr_change_delayed_minus_original": delayed_row[
                    "false_command_rate_per_rest_opportunity"
                ]
                - original_row["false_command_rate_per_rest_opportunity"],
            }
        )
    return pd.DataFrame(rows)


def build_channel_sensitivity(run_root: Path) -> pd.DataFrame:
    aggregate = pd.read_csv(
        run_root
        / "validation"
        / "subject_level_policy_validation"
        / "aggregate_summary.csv"
    )
    regions = pd.read_csv(
        run_root / "validation" / "channel_region_audit" / "policy_region_summary.csv"
    )
    region_index = regions.set_index(
        ["source", "policy", "budget", "m_per_class_per_run"]
    )
    rows: list[dict[str, object]] = []
    for mode in VALIDATION_MODES:
        agg = aggregate[aggregate["mode_label"] == mode].iloc[0]
        region_key = MODE_TO_REGION_KEY[mode]
        region = region_index.loc[region_key]
        rows.append(
            {
                "mode_label": mode,
                "display_name": agg["display_name"],
                "deployment_rate": agg["deployment_rate"],
                "deployed_actual_coverage": agg["deployed_actual_coverage"],
                "accepted_balanced_accuracy": agg["accepted_balanced_accuracy"],
                "unconditional_false_command_rate": agg[
                    "unconditional_false_command_rate"
                ],
                "unconditional_actual_coverage": agg["unconditional_actual_coverage"],
                "mean_deployed_channel_budget": agg["mean_deployed_channel_budget"],
                "mean_deployed_calibration_trials": agg[
                    "mean_deployed_calibration_trials"
                ],
                "motor_set_fraction": region["mean_motor_set_fraction"],
                "posterior_fraction": region["mean_posterior_fraction"],
                "frontal_fraction": region["mean_frontal_fraction"],
            }
        )
    return pd.DataFrame(rows)


def build_decision_matrix(
    window_df: pd.DataFrame, channel_df: pd.DataFrame
) -> pd.DataFrame:
    topk16 = channel_df[channel_df["mode_label"] == "topk16_m15_no_gate"].iloc[0]
    primary21 = channel_df[
        channel_df["mode_label"] == "topk21_m15_rule_t03_ba065_fcr010"
    ].iloc[0]
    minimal16 = channel_df[
        channel_df["mode_label"] == "topk16_m15_rule_t03_ba070_fcr010"
    ].iloc[0]
    meta21 = channel_df[channel_df["mode_label"] == "topk21_m15_t03_meta_p060"].iloc[0]
    motor21 = channel_df[channel_df["mode_label"] == "fixed_motor21_m15_no_gate"].iloc[0]
    topk16_window = window_df[
        (window_df["policy"] == "topk_effect") & (window_df["budget"] == 16)
    ].iloc[0]
    all_window = window_df[(window_df["policy"] == "all") & (window_df["budget"] == 64)].iloc[0]
    rows = [
        {
            "robustness_question": "Does the primary early-window evaluation support budgeted top-k?",
            "evidence_metric": "TopK21 primary BA and FCR",
            "value": f"BA={primary21['accepted_balanced_accuracy']:.3f}, FCR={primary21['unconditional_false_command_rate']:.3f}",
            "decision": "yes, with a readiness-gated safety/access tradeoff",
        },
        {
            "robustness_question": "Does the early window improve the operating frontier?",
            "evidence_metric": "BA change from 0.5-3.5 to 1.5-4.0",
            "value": f"All64 {all_window['ba_change_delayed_minus_original']:.3f}; TopK16 {topk16_window['ba_change_delayed_minus_original']:.3f}",
            "decision": "yes; use early window as v2 primary and delayed window as conservative sensitivity",
        },
        {
            "robustness_question": "Does the primary policy reduce false-command exposure?",
            "evidence_metric": "Primary21 unconditional FCR",
            "value": f"{primary21['unconditional_false_command_rate']:.3f}",
            "decision": "yes, with moderate deployment and reduced command coverage",
        },
        {
            "robustness_question": "Is the top-k result pure motor-channel decoding?",
            "evidence_metric": "TopK16 motor/posterior/frontal fractions",
            "value": f"motor={minimal16['motor_set_fraction']:.3f}, posterior={minimal16['posterior_fraction']:.3f}, frontal={minimal16['frontal_fraction']:.3f}",
            "decision": "no; frame as cue-paced command gating",
        },
        {
            "robustness_question": "Is fixed motor-only selection competitive?",
            "evidence_metric": "Motor21 accepted BA and FCR",
            "value": f"BA={motor21['accepted_balanced_accuracy']:.3f}, FCR={motor21['unconditional_false_command_rate']:.3f}",
            "decision": "no; keep as sensitivity baseline",
        },
        {
            "robustness_question": "Should the meta-readiness model be central?",
            "evidence_metric": "Meta21 deployment and unconditional FCR",
            "value": f"deployment={meta21['deployment_rate']:.3f}, FCR={meta21['unconditional_false_command_rate']:.3f}",
            "decision": "no; use as high-confidence secondary mode",
        },
    ]
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    window_df: pd.DataFrame,
    channel_df: pd.DataFrame,
    decision_df: pd.DataFrame,
) -> None:
    rows: list[str] = [
        "# Robustness Summary\n",
        f"Date: {report_date()}\n",
        "## Purpose\n",
        "Synthesize robustness checks already available in the project: time-window sensitivity, channel-region sensitivity, and operating-point hierarchy.\n",
        "## Window Sensitivity\n",
        "| Policy | Budget | BA 0.5-3.5 | BA 1.5-4.0 | Delta BA | FCR 0.5-3.5 | FCR 1.5-4.0 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in window_df.itertuples(index=False):
        rows.append(
            "| "
            f"{row.policy} | {int(row.budget)} | "
            f"{row.original_accepted_ba:.3f} | "
            f"{row.delayed_accepted_ba:.3f} | "
            f"{row.ba_change_delayed_minus_original:.3f} | "
            f"{row.original_fcr:.3f} | "
            f"{row.delayed_fcr:.3f} |"
        )
    rows.extend(
        [
            "\nInterpretation: the 0.5-3.5 s window gives the stronger operating frontier and is used as the v2 primary setting. The delayed 1.5-4.0 s window remains a conservative sensitivity check.",
            "\n## Channel-Region and Policy Sensitivity\n",
            "| Mode | Ch. | Deploy | BA | Uncond. FCR | Uncond. Cov. | Motor Frac. | Posterior Frac. | Frontal Frac. |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in channel_df.itertuples(index=False):
        rows.append(
            "| "
            f"{row.mode_label} | "
            f"{row.mean_deployed_channel_budget:.1f} | "
            f"{row.deployment_rate:.3f} | "
            f"{row.accepted_balanced_accuracy:.3f} | "
            f"{row.unconditional_false_command_rate:.3f} | "
            f"{row.unconditional_actual_coverage:.3f} | "
            f"{row.motor_set_fraction:.3f} | "
            f"{row.posterior_fraction:.3f} | "
            f"{row.frontal_fraction:.3f} |"
        )
    rows.extend(
        [
            "\n## Decision Matrix\n",
            "| Question | Evidence | Value | Decision |",
            "|---|---|---:|---|",
        ]
    )
    for row in decision_df.itertuples(index=False):
        rows.append(
            "| "
            f"{row.robustness_question} | "
            f"{row.evidence_metric} | "
            f"{row.value} | "
            f"{row.decision} |"
        )
    rows.extend(
        [
            "\n## Current Robust Conclusion\n",
            "The primary result should be presented as risk-aware cue-paced command gating, not pure motor-imagery decoding. The v2 early-window results support a 21-channel budgeted primary policy, a 16-channel minimal-burden secondary policy, and an interpretable readiness gate. The price of safety is lower unconditional command coverage than always-deploy baselines.",
        ]
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build robustness summary from existing validated outputs."
    )
    parser.add_argument("--config", default="configs/local.json")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    out_dir = cfg.run_root / "validation" / "robustness_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    window_df = build_window_sensitivity(cfg.run_root)
    channel_df = build_channel_sensitivity(cfg.run_root)
    decision_df = build_decision_matrix(window_df, channel_df)
    window_df.to_csv(out_dir / "window_sensitivity.csv", index=False)
    channel_df.to_csv(out_dir / "channel_policy_sensitivity.csv", index=False)
    decision_df.to_csv(out_dir / "robustness_decision_matrix.csv", index=False)
    write_report(out_dir / "robustness_summary.md", window_df, channel_df, decision_df)

    print(f"output_dir={out_dir}")
    print(f"window_sensitivity={out_dir / 'window_sensitivity.csv'}")
    print(f"channel_policy_sensitivity={out_dir / 'channel_policy_sensitivity.csv'}")
    print(f"robustness_decision_matrix={out_dir / 'robustness_decision_matrix.csv'}")
    print(decision_df.to_string(index=False))


if __name__ == "__main__":
    main()
