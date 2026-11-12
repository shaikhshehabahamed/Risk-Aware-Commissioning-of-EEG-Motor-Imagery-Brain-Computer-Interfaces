from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PROJECT_SRC))

from rabp_bmi.config import load_config
from rabp_bmi.quality import DEFAULT_EXCLUDED_SUBJECTS


TOLERANCE = 5e-6

EXPECTED_AGGREGATE_METRICS = {
    "all64_m15_no_gate": {
        "deployment_rate": 1.0,
        "deployed_actual_coverage": 0.3262433862433862,
        "accepted_balanced_accuracy": 0.8518255930844634,
        "unconditional_false_command_rate": 0.0463492063492063,
        "unconditional_actual_coverage": 0.3262433862433862,
    },
    "topk16_m15_no_gate": {
        "deployment_rate": 1.0,
        "deployed_actual_coverage": 0.2787301587301587,
        "accepted_balanced_accuracy": 0.8176531671858774,
        "unconditional_false_command_rate": 0.0507936507936507,
        "unconditional_actual_coverage": 0.2787301587301587,
    },
    "topk21_m15_no_gate": {
        "deployment_rate": 1.0,
        "deployed_actual_coverage": 0.2846560846560846,
        "accepted_balanced_accuracy": 0.8264720531267293,
        "unconditional_false_command_rate": 0.0484656084656084,
        "unconditional_actual_coverage": 0.2846560846560846,
    },
    "central21_m15_no_gate": {
        "deployment_rate": 1.0,
        "deployed_actual_coverage": 0.3108994708994709,
        "accepted_balanced_accuracy": 0.7916870828600489,
        "unconditional_false_command_rate": 0.0609523809523809,
        "unconditional_actual_coverage": 0.3108994708994709,
    },
    "fixed_motor21_m15_no_gate": {
        "deployment_rate": 1.0,
        "deployed_actual_coverage": 0.3051851851851851,
        "accepted_balanced_accuracy": 0.7845761849612962,
        "unconditional_false_command_rate": 0.0624338624338624,
        "unconditional_actual_coverage": 0.3051851851851851,
    },
    "topk21_m15_rule_t03_ba065_fcr010": {
        "deployment_rate": 0.5714285714285714,
        "deployed_actual_coverage": 0.2561111111111111,
        "accepted_balanced_accuracy": 0.8580523394730002,
        "unconditional_false_command_rate": 0.0162962962962962,
        "unconditional_actual_coverage": 0.1463492063492063,
    },
    "topk16_m15_rule_t03_ba070_fcr010": {
        "deployment_rate": 0.5492063492063493,
        "deployed_actual_coverage": 0.2549132947976879,
        "accepted_balanced_accuracy": 0.8518622337614594,
        "unconditional_false_command_rate": 0.016084656084656,
        "unconditional_actual_coverage": 0.14,
    },
    "topk16_m15_rule_ba075_fcr010": {
        "deployment_rate": 0.4126984126984127,
        "deployed_actual_coverage": 0.3630769230769231,
        "accepted_balanced_accuracy": 0.8550935908221644,
        "unconditional_false_command_rate": 0.0167195767195767,
        "unconditional_actual_coverage": 0.1498412698412698,
    },
    "topk21_m15_t03_meta_p060": {
        "deployment_rate": 0.1301587301587301,
        "deployed_actual_coverage": 0.3227642276422764,
        "accepted_balanced_accuracy": 0.9269899031315668,
        "unconditional_false_command_rate": 0.0033862433862433,
        "unconditional_actual_coverage": 0.042010582010582,
    },
}

EXPECTED_WINDOW_METRICS = {
    ("all", 64): {
        "original_accepted_ba": 0.8518255930844634,
        "delayed_accepted_ba": 0.7948683070470144,
        "ba_change_delayed_minus_original": -0.05695728603744899,
    },
    ("topk_effect", 16): {
        "original_accepted_ba": 0.8176531671858774,
        "delayed_accepted_ba": 0.8026309833723173,
        "ba_change_delayed_minus_original": -0.015022183813560083,
    },
    ("topk_effect", 21): {
        "original_accepted_ba": 0.8264720531267293,
        "delayed_accepted_ba": 0.7978525653783386,
        "ba_change_delayed_minus_original": -0.0286194877483907,
    },
    ("fixed_motor", 21): {
        "original_accepted_ba": 0.7845761849612962,
        "delayed_accepted_ba": 0.7516777880042951,
        "ba_change_delayed_minus_original": -0.03289839695700106,
    },
}

EXPECTED_CHANNEL_REGION = {
    ("calibration_limited", "topk_effect", 16, 15): {
        "mean_motor_set_fraction": 0.4093253968253968,
        "mean_posterior_fraction": 0.1676587301587301,
        "mean_frontal_fraction": 0.2418650793650793,
    },
    ("calibration_limited", "topk_effect", 21, 15): {
        "mean_motor_set_fraction": 0.4019652305366591,
        "mean_posterior_fraction": 0.1564625850340136,
        "mean_frontal_fraction": 0.2433862433862433,
    },
    ("calibration_limited", "central_parietal_topk_effect", 21, 15): {
        "mean_motor_set_fraction": 0.7253212396069539,
        "mean_posterior_fraction": 0.0,
        "mean_frontal_fraction": 0.0,
    },
    ("calibration_limited", "fixed_motor", 21, 15): {
        "mean_motor_set_fraction": 1.0,
        "mean_posterior_fraction": 0.0,
        "mean_frontal_fraction": 0.0,
    },
}

EXPECTED_MANIFEST_ARTIFACTS = (
    "README.md",
    "pyproject.toml",
    "scripts/build_project_manifest.py",
    "scripts/build_paper_results_package.py",
    "src/rabp_bmi/features.py",
    "audit/dataset_audit.csv",
    "budgeted_rejection/budgeted_logvar_intent_rest_policies_v2_t0.5-3.5/summary.csv",
    "budgeted_rejection/budgeted_logvar_intent_rest_policies_v2_t1.5-4/summary.csv",
    "calibration_frontier/calibration_frontier_logvar_intent_rest_t0.5-3.5/summary.csv",
    "calibration_frontier/calibration_frontier_logvar_intent_rest_t1.5-4/summary.csv",
    "readiness_gate/readiness_gate_logvar_intent_rest_t0.5-3.5_target_sweep/fold_results.csv",
    "readiness_gate/readiness_gate_logvar_intent_rest_t0.5-3.5_target_sweep/meta_readiness/meta_predictions.csv",
    "validation/subject_level_policy_validation/aggregate_summary.csv",
    "validation/subject_level_policy_validation/paired_comparisons.csv",
    "validation/subject_level_policy_validation/milestone_06_statistical_validation.md",
    "validation/channel_region_audit/policy_region_summary.csv",
    "validation/channel_region_audit/channel_region_audit.md",
    "validation/robustness_summary/window_sensitivity.csv",
    "validation/robustness_summary/robustness_summary.md",
    "final_protocol/analysis_protocol_v2.0/analysis_protocol.json",
    "final_protocol/analysis_protocol_v2.0/analysis_protocol.md",
    "final_protocol/analysis_protocol_v2.0/operating_point_hierarchy.csv",
    "paper_assets/operating_points/operating_point_table.csv",
    "paper_assets/operating_points/policy_tradeoff_unconditional_fcr_vs_ba.png",
    "quality_assurance/core_output_verification/verification_checks.csv",
    "quality_assurance/core_output_verification/verification_report.md",
)


def report_date() -> str:
    return datetime.now().date().isoformat()


def check_close(
    rows: list[dict[str, object]],
    *,
    check_id: str,
    observed: float,
    expected: float,
    tolerance: float = TOLERANCE,
) -> None:
    diff = abs(float(observed) - float(expected))
    rows.append(
        {
            "check_id": check_id,
            "observed": observed,
            "expected": expected,
            "tolerance": tolerance,
            "absolute_difference": diff,
            "status": "PASS" if diff <= tolerance else "FAIL",
        }
    )


def check_equal(
    rows: list[dict[str, object]],
    *,
    check_id: str,
    observed: object,
    expected: object,
) -> None:
    rows.append(
        {
            "check_id": check_id,
            "observed": observed,
            "expected": expected,
            "tolerance": 0,
            "absolute_difference": 0 if observed == expected else "NA",
            "status": "PASS" if observed == expected else "FAIL",
        }
    )


def check_at_least(
    rows: list[dict[str, object]],
    *,
    check_id: str,
    observed: int,
    minimum: int,
) -> None:
    rows.append(
        {
            "check_id": check_id,
            "observed": observed,
            "expected": f">={minimum}",
            "tolerance": 0,
            "absolute_difference": max(0, minimum - observed),
            "status": "PASS" if observed >= minimum else "FAIL",
        }
    )


def normalize_manifest_path(value: object) -> str:
    return str(value).replace("\\", "/")


def write_report(path: Path, checks: pd.DataFrame) -> None:
    failed = checks[checks["status"] != "PASS"]
    rows = [
        "# Core Output Verification\n",
        f"Date: {report_date()}\n",
        "## Summary\n",
        "```text",
        f"checks: {len(checks)}",
        f"passed: {(checks['status'] == 'PASS').sum()}",
        f"failed: {len(failed)}",
        "```",
        "## Failed Checks\n",
    ]
    if failed.empty:
        rows.append("No failed checks.")
    else:
        rows.append("| Check | Observed | Expected | Tolerance |")
        rows.append("|---|---:|---:|---:|")
        for row in failed.itertuples(index=False):
            rows.append(
                f"| {row.check_id} | {row.observed} | {row.expected} | {row.tolerance} |"
            )
    rows.extend(
        [
            "\n## Interpretation\n",
            "This verifier is intentionally narrow. It checks the core validated outputs and key project metadata. A pass means the current saved outputs match the validated state; it does not replace rerunning the full EEG-processing pipeline.",
        ]
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify core saved outputs against validated expected values."
    )
    parser.add_argument("--config", default="configs/local.json")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    rows: list[dict[str, object]] = []

    audit = pd.read_csv(cfg.run_root / "audit" / "dataset_audit.csv")
    target = audit[audit["run"].isin(cfg.task_runs)]
    subjects = set(target["subject"].unique())
    clean_subjects = subjects - set(DEFAULT_EXCLUDED_SUBJECTS)
    check_equal(
        rows,
        check_id="dataset.target_run_files",
        observed=len(target),
        expected=327,
    )
    check_equal(
        rows,
        check_id="dataset.target_subjects",
        observed=len(subjects),
        expected=109,
    )
    check_equal(
        rows,
        check_id="dataset.clean_subjects",
        observed=len(clean_subjects),
        expected=105,
    )

    aggregate = pd.read_csv(
        cfg.run_root
        / "validation"
        / "subject_level_policy_validation"
        / "aggregate_summary.csv"
    ).set_index("mode_label")
    for mode, metrics in EXPECTED_AGGREGATE_METRICS.items():
        for metric, expected in metrics.items():
            check_close(
                rows,
                check_id=f"aggregate.{mode}.{metric}",
                observed=float(aggregate.loc[mode, metric]),
                expected=expected,
            )

    window = pd.read_csv(
        cfg.run_root / "validation" / "robustness_summary" / "window_sensitivity.csv"
    ).set_index(["policy", "budget"])
    for key, metrics in EXPECTED_WINDOW_METRICS.items():
        for metric, expected in metrics.items():
            check_close(
                rows,
                check_id=f"window.{key[0]}_k{key[1]}.{metric}",
                observed=float(window.loc[key, metric]),
                expected=expected,
            )

    regions = pd.read_csv(
        cfg.run_root / "validation" / "channel_region_audit" / "policy_region_summary.csv"
    ).set_index(["source", "policy", "budget", "m_per_class_per_run"])
    for key, metrics in EXPECTED_CHANNEL_REGION.items():
        for metric, expected in metrics.items():
            check_close(
                rows,
                check_id=f"regions.{key[1]}_k{key[2]}_m{key[3]}.{metric}",
                observed=float(regions.loc[key, metric]),
                expected=expected,
            )

    manifest_path = cfg.run_root / "project_manifest" / "project_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        check_equal(
            rows,
            check_id="manifest.claim_count",
            observed=int(manifest["claim_count"]),
            expected=10,
        )
        check_at_least(
            rows,
            check_id="manifest.artifact_count_minimum",
            observed=int(manifest["artifact_count"]),
            minimum=125,
        )
        artifact_index_path = cfg.run_root / "project_manifest" / "artifact_index.csv"
        if artifact_index_path.exists():
            artifact_index = pd.read_csv(artifact_index_path)
            manifest_paths = {
                normalize_manifest_path(path)
                for path in artifact_index["relative_path"].tolist()
            }
            for expected_path in EXPECTED_MANIFEST_ARTIFACTS:
                check_equal(
                    rows,
                    check_id=f"manifest.artifact_present.{expected_path}",
                    observed=expected_path in manifest_paths,
                    expected=True,
                )
        else:
            check_equal(
                rows,
                check_id="manifest.artifact_index.exists",
                observed=False,
                expected=True,
            )
    else:
        check_equal(
            rows,
            check_id="manifest.exists",
            observed=False,
            expected=True,
        )

    checks = pd.DataFrame(rows)
    out_dir = cfg.run_root / "quality_assurance" / "core_output_verification"
    out_dir.mkdir(parents=True, exist_ok=True)
    checks.to_csv(out_dir / "verification_checks.csv", index=False)
    write_report(out_dir / "verification_report.md", checks)

    n_failed = int((checks["status"] != "PASS").sum())
    print(f"output_dir={out_dir}")
    print(f"checks={len(checks)} passed={len(checks) - n_failed} failed={n_failed}")
    if n_failed:
        print(checks[checks["status"] != "PASS"].to_string(index=False))
    if args.fail_on_error and n_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
