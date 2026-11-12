from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PROJECT_SRC))

from rabp_bmi.config import load_config
from rabp_bmi.quality import DEFAULT_EXCLUDED_SUBJECTS


PROTOCOL_VERSION = "v2.0"


def report_date() -> str:
    return datetime.now().date().isoformat()


OPERATING_POINT_ORDER = [
    "topk21_m15_rule_t03_ba065_fcr010",
    "topk16_m15_rule_t03_ba070_fcr010",
    "all64_m15_no_gate",
    "topk16_m15_no_gate",
    "topk21_m15_no_gate",
    "topk16_m15_rule_ba075_fcr010",
    "topk21_m15_t03_meta_p060",
    "central21_m15_no_gate",
    "fixed_motor21_m15_no_gate",
]

OPERATING_POINT_ROLES = {
    "all64_m15_no_gate": "primary_comparator",
    "topk16_m15_no_gate": "always_deploy_budgeted_baseline",
    "topk21_m15_no_gate": "secondary_budgeted_baseline",
    "central21_m15_no_gate": "biological_sensitivity",
    "fixed_motor21_m15_no_gate": "motor_only_negative_sensitivity",
    "topk21_m15_rule_t03_ba065_fcr010": "primary_policy",
    "topk16_m15_rule_t03_ba070_fcr010": "minimal_channel_secondary",
    "topk16_m15_rule_ba075_fcr010": "safety_secondary",
    "topk21_m15_t03_meta_p060": "high_confidence_secondary",
}


def metric_ci(ci_df: pd.DataFrame, mode_label: str, metric: str) -> str:
    row = ci_df[(ci_df["mode_label"] == mode_label) & (ci_df["metric"] == metric)]
    if row.empty:
        return ""
    item = row.iloc[0]
    return (
        f"{float(item['point_estimate']):.3f} "
        f"[{float(item['ci_low']):.3f}, {float(item['ci_high']):.3f}]"
    )


def build_operating_hierarchy(run_root: Path) -> pd.DataFrame:
    aggregate = pd.read_csv(
        run_root
        / "validation"
        / "subject_level_policy_validation"
        / "aggregate_summary.csv"
    ).set_index("mode_label")
    ci_df = pd.read_csv(
        run_root
        / "validation"
        / "subject_level_policy_validation"
        / "bootstrap_confidence_intervals.csv"
    )
    rows: list[dict[str, object]] = []
    for rank, mode_label in enumerate(OPERATING_POINT_ORDER, start=1):
        agg = aggregate.loc[mode_label]
        rows.append(
            {
                "rank": rank,
                "mode_label": mode_label,
                "role": OPERATING_POINT_ROLES[mode_label],
                "display_name": agg["display_name"],
                "mean_channels": agg["mean_deployed_channel_budget"],
                "mean_calibration_trials": agg["mean_deployed_calibration_trials"],
                "deployment_rate": agg["deployment_rate"],
                "deployment_rate_ci": metric_ci(ci_df, mode_label, "deployment_rate"),
                "deployed_coverage": agg["deployed_actual_coverage"],
                "deployed_coverage_ci": metric_ci(
                    ci_df, mode_label, "deployed_actual_coverage"
                ),
                "accepted_balanced_accuracy": agg["accepted_balanced_accuracy"],
                "accepted_balanced_accuracy_ci": metric_ci(
                    ci_df, mode_label, "accepted_balanced_accuracy"
                ),
                "deployed_false_command_rate": agg[
                    "false_command_rate_per_rest_opportunity"
                ],
                "deployed_false_command_rate_ci": metric_ci(
                    ci_df,
                    mode_label,
                    "false_command_rate_per_rest_opportunity",
                ),
                "unconditional_coverage": agg["unconditional_actual_coverage"],
                "unconditional_coverage_ci": metric_ci(
                    ci_df, mode_label, "unconditional_actual_coverage"
                ),
                "unconditional_false_command_rate": agg[
                    "unconditional_false_command_rate"
                ],
                "unconditional_false_command_rate_ci": metric_ci(
                    ci_df, mode_label, "unconditional_false_command_rate"
                ),
            }
        )
    return pd.DataFrame(rows)


def endpoint_definitions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "endpoint": "accepted_balanced_accuracy",
                "role": "primary_reliability_endpoint",
                "definition": "Balanced accuracy computed only on accepted trials in deployed sessions.",
                "direction": "higher_is_better",
            },
            {
                "endpoint": "unconditional_false_command_rate",
                "role": "primary_safety_endpoint",
                "definition": "False intent commands divided by all rest opportunities, including no-go sessions.",
                "direction": "lower_is_better",
            },
            {
                "endpoint": "deployment_rate",
                "role": "access_endpoint",
                "definition": "Held-out subject-run folds accepted for deployment by the policy.",
                "direction": "higher_is_better",
            },
            {
                "endpoint": "unconditional_actual_coverage",
                "role": "access_and_utility_endpoint",
                "definition": "Accepted held-out trials divided by all held-out trials, including no-go sessions.",
                "direction": "higher_is_better",
            },
            {
                "endpoint": "deployed_actual_coverage",
                "role": "conditional_utility_endpoint",
                "definition": "Accepted held-out trials divided by held-out trials only among deployed sessions.",
                "direction": "higher_is_better",
            },
            {
                "endpoint": "mean_deployed_channel_budget",
                "role": "resource_endpoint",
                "definition": "Mean number of EEG channels required when the policy deploys.",
                "direction": "lower_is_better",
            },
            {
                "endpoint": "mean_deployed_calibration_trials",
                "role": "resource_endpoint",
                "definition": "Mean number of labeled calibration trials required when the policy deploys.",
                "direction": "lower_is_better",
            },
        ]
    )


def inclusion_exclusion_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rule": "dataset",
                "decision": "include",
                "details": "PhysioNet EEG Motor Movement/Imagery Database; target runs R04, R08, R12.",
            },
            {
                "rule": "task",
                "decision": "include",
                "details": "Cue-paced rest-vs-intent command gating: T0=rest, T1/T2=intent.",
            },
            {
                "rule": "primary_window",
                "decision": "include",
                "details": "Use 0.5-3.5 s post-cue analysis window as final v2 primary setting.",
            },
            {
                "rule": "known_problematic_subjects",
                "decision": "exclude_primary",
                "details": "Exclude "
                + ", ".join(sorted(DEFAULT_EXCLUDED_SUBJECTS))
                + " due to sampling-rate mismatch, damaged/short recordings, or known usability concerns.",
            },
            {
                "rule": "delayed_window",
                "decision": "sensitivity_only",
                "details": "1.5-4.0 s window is retained as a conservative delayed-window sensitivity analysis.",
            },
            {
                "rule": "left_right_motor_imagery",
                "decision": "not_primary",
                "details": "Left-vs-right motor imagery was evaluated and found too weak for the primary OR decision-analytics contribution.",
            },
        ]
    )


def analysis_decisions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "decision_id": "D01",
                "decision": "Primary scientific framing",
                "locked_choice": "Risk-aware operations-research framework for cue-paced EEG command gating.",
                "rationale": "This matches the validated evidence and avoids overclaiming pure motor decoding.",
            },
            {
                "decision_id": "D02",
                "decision": "Primary operating policy",
                "locked_choice": "Primary21: top-k 21 channels, 60 calibration trials, target coverage 0.3, deploy if calibration BA >= 0.65 and FCR <= 0.10.",
                "rationale": "Strongest validated budgeted reliability-safety-access tradeoff after the v2 window and target sweep.",
            },
            {
                "decision_id": "D03",
                "decision": "Always-deploy budgeted baseline",
                "locked_choice": "TopK16 no-gate.",
                "rationale": "Shows resource reduction relative to all channels without readiness gating.",
            },
            {
                "decision_id": "D04",
                "decision": "Primary comparator",
                "locked_choice": "All64 no-gate.",
                "rationale": "Conventional full-channel benchmark.",
            },
            {
                "decision_id": "D05",
                "decision": "High-confidence secondary",
                "locked_choice": "TopK16 target0.4 Rule75 and TopK21 target0.3 Meta21.",
                "rationale": "Useful sensitivity modes for high-confidence operation, but not the best primary deployment/coverage tradeoff.",
            },
            {
                "decision_id": "D06",
                "decision": "Biological sensitivity",
                "locked_choice": "Central21 and Motor21.",
                "rationale": "Quantifies what remains when posterior/frontal or non-motor channels are constrained.",
            },
            {
                "decision_id": "D07",
                "decision": "Statistical validation",
                "locked_choice": "Subject-level bootstrap confidence intervals and paired subject-level comparisons.",
                "rationale": "Preserves subject heterogeneity and avoids trial-level pseudo-replication.",
            },
            {
                "decision_id": "D08",
                "decision": "Non-inferiority interpretation",
                "locked_choice": "Use 0.05 BA margin only as a screening/engineering margin.",
                "rationale": "No clinical margin has been externally established for this dataset and task.",
            },
            {
                "decision_id": "D09",
                "decision": "Protocol amendment",
                "locked_choice": "v2.0 supersedes v1.0 after exploratory operating-point and window audits.",
                "rationale": "The 0.5-3.5 s window improved the validated budgeted policy frontier; this amendment records the data-driven pivot explicitly.",
            },
        ]
    )


def out_of_scope_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item": "new_classifier_claim",
                "status": "out_of_scope",
                "reason": "Novelty is in decision framework, budget formulation, evaluation design, and deployment policy.",
            },
            {
                "item": "pure_motor_cortex_decoding_claim",
                "status": "out_of_scope",
                "reason": "Top-k policies select meaningful frontal/posterior fractions; cue-paced command-gating framing is more honest.",
            },
            {
                "item": "self_paced_asynchronous_bmi_claim",
                "status": "out_of_scope",
                "reason": "Dataset and labels support cue-paced analysis, not self-paced asynchronous command detection.",
            },
            {
                "item": "universal_deployment_claim",
                "status": "out_of_scope",
                "reason": "Readiness gates deliberately reject many sessions; lower-tail subjects remain a limitation.",
            },
            {
                "item": "clinical_effectiveness_claim",
                "status": "out_of_scope",
                "reason": "The public dataset supports methodological validation, not clinical efficacy.",
            },
        ]
    )


def protocol_json(
    *,
    cfg,
    operating_hierarchy: pd.DataFrame,
) -> dict[str, object]:
    primary = operating_hierarchy[
        operating_hierarchy["role"] == "primary_policy"
    ].iloc[0]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_v2_for_reproducible_analysis_not_manuscript_submission",
        "dataset": {
            "name": "PhysioNet EEG Motor Movement/Imagery Database",
            "dataset_root": str(cfg.dataset_root),
            "task_runs": list(cfg.task_runs),
            "primary_clean_subject_count": 105,
            "excluded_subjects_primary": sorted(DEFAULT_EXCLUDED_SUBJECTS),
        },
        "task": {
            "name": "cue_paced_rest_vs_intent_command_gating",
            "rest_label": "T0",
            "intent_labels": ["T1", "T2"],
            "primary_window_seconds": [0.5, 3.5],
            "sensitivity_window_seconds": [1.5, 4.0],
        },
        "primary_policy": {
            "mode_label": primary["mode_label"],
            "description": primary["display_name"],
            "channels": float(primary["mean_channels"]),
            "calibration_trials": float(primary["mean_calibration_trials"]),
            "readiness_rule": "target coverage 0.3; deploy if calibration accepted BA >= 0.65 and calibration FCR <= 0.10",
        },
        "primary_endpoints": [
            "accepted_balanced_accuracy",
            "unconditional_false_command_rate",
            "deployment_rate",
            "unconditional_actual_coverage",
        ],
        "primary_validation": [
            "subject_level_bootstrap_confidence_intervals",
            "paired_subject_level_comparisons",
            "core_output_verification",
            "claim_evidence_manifest",
        ],
        "change_control": {
            "rule": "Any further change to primary task, primary window, primary policy, cohort, or endpoint hierarchy must create a new protocol version with rationale.",
            "next_version_if_changed": "v3.0",
            "v2_amendment_rationale": "Exploratory window, target-coverage, and classifier audits found the strongest budgeted operating frontier in the 0.5-3.5 s logistic-readiness setting.",
        },
    }


def write_markdown_report(
    path: Path,
    protocol: dict[str, object],
    hierarchy: pd.DataFrame,
    decisions: pd.DataFrame,
    out_of_scope: pd.DataFrame,
) -> None:
    primary = hierarchy[hierarchy["role"] == "primary_policy"].iloc[0]
    rows: list[str] = [
        f"# Frozen Analysis Protocol {PROTOCOL_VERSION}\n",
        f"Date: {report_date()}\n",
        "## Status\n",
        f"`{protocol['status']}`\n",
        "This is not a manuscript draft. It is the locked analysis protocol for the next reproducibility and paper-preparation phase.\n",
        "This v2.0 protocol supersedes v1.0 after an explicitly recorded exploratory audit of target coverage, analysis window, and classifier alternatives.\n",
        "## Primary Research Claim\n",
        "A risk-aware, budgeted, calibration-dependent operations-research framework can support cue-paced EEG command gating by explicitly trading channel burden, calibration burden, deployment, command coverage, accepted-command reliability, and false-command exposure.\n",
        "## Primary Policy\n",
        "```text",
        f"{primary['mode_label']}",
        "0.5-3.5 s post-cue window",
        "top-k 21 channels, 60 calibration trials",
        "target coverage 0.3",
        "deploy if calibration accepted BA >= 0.65 and calibration FCR <= 0.10",
        "```",
        "Primary validated operating point:",
        "```text",
        f"deployment rate: {primary['deployment_rate_ci']}",
        f"accepted BA: {primary['accepted_balanced_accuracy_ci']}",
        f"unconditional FCR: {primary['unconditional_false_command_rate_ci']}",
        f"unconditional coverage: {primary['unconditional_coverage_ci']}",
        "```",
        "## Operating-Point Hierarchy\n",
        "| Rank | Mode | Role | Channels | Trials | Deploy | BA | Uncond. FCR | Uncond. Cov. |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in hierarchy.itertuples(index=False):
        rows.append(
            "| "
            f"{int(row.rank)} | {row.mode_label} | {row.role} | "
            f"{float(row.mean_channels):.1f} | "
            f"{float(row.mean_calibration_trials):.1f} | "
            f"{row.deployment_rate_ci} | "
            f"{row.accepted_balanced_accuracy_ci} | "
            f"{row.unconditional_false_command_rate_ci} | "
            f"{row.unconditional_coverage_ci} |"
        )
    rows.extend(
        [
            "\n## Locked Decisions\n",
            "| ID | Decision | Locked Choice | Rationale |",
            "|---|---|---|---|",
        ]
    )
    for row in decisions.itertuples(index=False):
        rows.append(
            f"| {row.decision_id} | {row.decision} | {row.locked_choice} | {row.rationale} |"
        )
    rows.extend(
        [
            "\n## Out Of Scope Claims\n",
            "| Item | Status | Reason |",
            "|---|---|---|",
        ]
    )
    for row in out_of_scope.itertuples(index=False):
        rows.append(f"| {row.item} | {row.status} | {row.reason} |")
    rows.extend(
        [
            "\n## Change Control\n",
            "Any change to the primary cohort, task definition, primary time window, primary operating policy, endpoint hierarchy, or statistical validation plan must create a new protocol version with a rationale.",
        ]
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze final analysis protocol and operating-point hierarchy."
    )
    parser.add_argument("--config", default="configs/local.json")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    out_dir = cfg.run_root / "final_protocol" / f"analysis_protocol_{PROTOCOL_VERSION}"
    out_dir.mkdir(parents=True, exist_ok=True)

    hierarchy = build_operating_hierarchy(cfg.run_root)
    endpoints = endpoint_definitions()
    inclusion = inclusion_exclusion_table()
    decisions = analysis_decisions()
    out_scope = out_of_scope_table()
    protocol = protocol_json(cfg=cfg, operating_hierarchy=hierarchy)

    hierarchy.to_csv(out_dir / "operating_point_hierarchy.csv", index=False)
    endpoints.to_csv(out_dir / "endpoint_definitions.csv", index=False)
    inclusion.to_csv(out_dir / "inclusion_exclusion_rules.csv", index=False)
    decisions.to_csv(out_dir / "locked_analysis_decisions.csv", index=False)
    out_scope.to_csv(out_dir / "out_of_scope_claims.csv", index=False)
    with (out_dir / "analysis_protocol.json").open("w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2)
    write_markdown_report(
        out_dir / "analysis_protocol.md",
        protocol=protocol,
        hierarchy=hierarchy,
        decisions=decisions,
        out_of_scope=out_scope,
    )

    print(f"output_dir={out_dir}")
    print(f"protocol={out_dir / 'analysis_protocol.md'}")
    print(f"hierarchy={out_dir / 'operating_point_hierarchy.csv'}")
    print(
        hierarchy[
            [
                "rank",
                "mode_label",
                "role",
                "deployment_rate",
                "accepted_balanced_accuracy",
                "unconditional_false_command_rate",
                "unconditional_coverage",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )


if __name__ == "__main__":
    main()
