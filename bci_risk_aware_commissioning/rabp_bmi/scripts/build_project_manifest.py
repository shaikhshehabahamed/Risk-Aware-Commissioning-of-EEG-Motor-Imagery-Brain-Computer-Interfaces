from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import sys

import pandas as pd

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PROJECT_SRC))

from rabp_bmi.config import load_config
from rabp_bmi.quality import DEFAULT_EXCLUDED_SUBJECTS


PACKAGE_NAMES = [
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "mne",
    "matplotlib",
]

CORE_RESULT_PATTERNS = [
    "audit/*.csv",
    "budgeted_rejection/budgeted_logvar_intent_rest_policies_v2_t0.5-3.5/*.csv",
    "budgeted_rejection/budgeted_logvar_intent_rest_policies_v2_t1.5-4/*.csv",
    "calibration_frontier/calibration_frontier_logvar_intent_rest_t0.5-3.5/*.csv",
    "calibration_frontier/calibration_frontier_logvar_intent_rest_t1.5-4/*.csv",
    "readiness_gate/readiness_gate_logvar_intent_rest_t0.5-3.5_target_sweep/*.csv",
    "readiness_gate/readiness_gate_logvar_intent_rest_t0.5-3.5_target_sweep/meta_readiness/*.csv",
    "readiness_gate/readiness_gate_logvar_intent_rest_t1.5-4/*.csv",
    "readiness_gate/readiness_gate_logvar_intent_rest_t1.5-4/meta_readiness/*.csv",
    "validation/**/*.csv",
    "validation/**/*.md",
    "final_protocol/**/*.csv",
    "final_protocol/**/*.json",
    "final_protocol/**/*.md",
    "quality_assurance/**/*.csv",
    "quality_assurance/**/*.md",
    "paper_assets/**/*.csv",
    "paper_assets/**/*.md",
    "paper_assets/**/*.png",
    "paper_assets/**/*.pdf",
]

CLAIMS = [
    {
        "claim_id": "C01_scope_pivot",
        "claim": "Left-vs-right motor imagery was weak; the project pivoted to cue-paced rest-vs-intent command gating.",
        "evidence": "reports/milestone_01_data_audit_and_baselines.md",
        "status": "supported",
    },
    {
        "claim_id": "C02_clean_cohort",
        "claim": "Primary analyses use a 105-subject clean cohort after excluding known problematic subjects.",
        "evidence": "reports/milestone_01_data_audit_and_baselines.md; ../rabp_bmi_runs/audit/dataset_audit.csv",
        "status": "supported",
    },
    {
        "claim_id": "C03_topk_noninferiority",
        "claim": "Top-k 16/21 channel policies are balanced-accuracy non-inferior to all 64 channels under a 0.05 screening margin.",
        "evidence": "reports/milestone_06_statistical_validation.md; ../rabp_bmi_runs/validation/subject_level_policy_validation/paired_comparisons.csv",
        "status": "supported_as_screening_noninferiority",
    },
    {
        "claim_id": "C04_v2_primary",
        "claim": "The final v2 primary operating point is a 21-channel rule readiness gate in the 0.5-3.5 s window with target coverage 0.3, calibration BA >= 0.65, and calibration FCR <= 0.10.",
        "evidence": "reports/milestone_08_operating_point_assets.md; ../rabp_bmi_runs/paper_assets/operating_points/operating_point_table.csv",
        "status": "supported",
    },
    {
        "claim_id": "C05_v2_primary_tradeoff",
        "claim": "The v2 primary readiness gate sharply reduces unconditional false-command exposure relative to always-deploy baselines, but lowers unconditional command coverage.",
        "evidence": "reports/milestone_06_statistical_validation.md; ../rabp_bmi_runs/validation/subject_level_policy_validation/bootstrap_confidence_intervals.csv",
        "status": "supported",
    },
    {
        "claim_id": "C06_meta_secondary",
        "claim": "Meta-readiness is useful as a high-confidence sensitivity mode but does not dominate the simple readiness rule.",
        "evidence": "reports/milestone_05_meta_readiness_model.md; reports/milestone_08_operating_point_assets.md",
        "status": "supported_secondary",
    },
    {
        "claim_id": "C07_channel_region",
        "claim": "Unconstrained top-k selection is not pure motor-channel decoding and must be framed as cue-paced command gating.",
        "evidence": "reports/milestone_07_channel_region_audit.md; ../rabp_bmi_runs/validation/channel_region_audit/policy_region_summary.csv",
        "status": "supported_with_framing_constraint",
    },
    {
        "claim_id": "C08_fixed_motor_weaker",
        "claim": "Fixed motor-only channels are weaker than top-k and should be a sensitivity baseline, not the main policy.",
        "evidence": "reports/milestone_06_statistical_validation.md; ../rabp_bmi_runs/validation/subject_level_policy_validation/aggregate_summary.csv",
        "status": "supported",
    },
    {
        "claim_id": "C09_early_window_primary",
        "claim": "The 0.5-3.5 s window is the final v2 primary analysis window because it provides the strongest validated budgeted operating frontier; the delayed 1.5-4.0 s window remains a conservative sensitivity check.",
        "evidence": "reports/milestone_10_robustness_summary.md; ../rabp_bmi_runs/validation/robustness_summary/window_sensitivity.csv",
        "status": "supported_as_protocol_v2_amendment",
    },
    {
        "claim_id": "C10_or_decision_framework",
        "claim": "The robust contribution is a risk-aware operations-research decision framework, not a claim of a new classifier or pure motor-cortex decoder.",
        "evidence": "reports/milestone_10_robustness_summary.md; ../rabp_bmi_runs/project_manifest/claim_evidence_register.md",
        "status": "supported_as_project_framing",
    },
]


def report_date() -> str:
    return datetime.now().date().isoformat()


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def csv_shape(path: Path) -> tuple[int | None, int | None]:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return 0, 0
            row_count = sum(1 for _ in reader)
            return row_count, len(header)
    except UnicodeDecodeError:
        return None, None


def collect_project_files(project_root: Path) -> list[Path]:
    paths: list[Path] = []
    paths.extend(
        [
            project_root / "README.md",
            project_root / "pyproject.toml",
            project_root / "requirements.txt",
        ]
    )
    paths.extend((project_root / "configs").glob("*.json"))
    paths.extend((project_root / "src").rglob("*.py"))
    paths.extend((project_root / "scripts").glob("*.py"))
    paths.extend((project_root / "reports").glob("*.md"))
    return sorted({p.resolve() for p in paths if p.exists()})


def collect_result_files(run_root: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in CORE_RESULT_PATTERNS:
        paths.extend(run_root.glob(pattern))
    return sorted({p.resolve() for p in paths if p.is_file()})


def artifact_row(path: Path, *, root: Path, category: str) -> dict[str, object]:
    row_count, column_count = csv_shape(path) if path.suffix.lower() == ".csv" else (None, None)
    stat = path.stat()
    return {
        "category": category,
        "relative_path": str(path.relative_to(root)),
        "suffix": path.suffix.lower(),
        "bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, timezone.utc
        ).isoformat(),
        "sha256": sha256_file(path),
        "csv_rows": row_count,
        "csv_columns": column_count,
    }


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in PACKAGE_NAMES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def dataset_summary(audit_csv: Path) -> dict[str, object]:
    if not audit_csv.exists():
        return {"audit_available": False}
    df = pd.read_csv(audit_csv)
    target = df[df["run"].isin([4, 8, 12])].copy()
    subjects = sorted(target["subject"].unique().tolist())
    excluded = sorted(DEFAULT_EXCLUDED_SUBJECTS)
    clean = [subject for subject in subjects if subject not in DEFAULT_EXCLUDED_SUBJECTS]
    sampling_rates = (
        target.groupby("sfreq")
        .size()
        .reset_index(name="file_count")
        .to_dict(orient="records")
    )
    return {
        "audit_available": True,
        "target_run_files": int(len(target)),
        "target_subjects": int(len(subjects)),
        "excluded_subjects": excluded,
        "clean_subjects": int(len(clean)),
        "task_runs": [4, 8, 12],
        "sampling_rate_file_counts": sampling_rates,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_command_sequence(path: Path) -> None:
    commands = [
        "# Reproducibility command sequence for the current validated project state.",
        "# Run from the rabp_bmi project folder.",
        "",
        ".\\.venv\\Scripts\\python -m pip install -r requirements.txt",
        ".\\.venv\\Scripts\\python -m pip install -e .",
        ".\\.venv\\Scripts\\python scripts\\audit_dataset.py --config configs\\local.json --max-read 327",
        ".\\.venv\\Scripts\\python scripts\\run_budgeted_logvar_rejection.py --config configs\\local.json --tmin 0.5 --tmax 3.5",
        ".\\.venv\\Scripts\\python scripts\\run_budgeted_logvar_rejection.py --config configs\\local.json --tmin 1.5 --tmax 4.0",
        ".\\.venv\\Scripts\\python scripts\\run_calibration_budget_frontier.py --config configs\\local.json --tmin 0.5 --tmax 3.5",
        ".\\.venv\\Scripts\\python scripts\\summarize_calibration_frontier.py ..\\rabp_bmi_runs\\calibration_frontier\\calibration_frontier_logvar_intent_rest_t0.5-3.5",
        ".\\.venv\\Scripts\\python scripts\\recommend_frontier_policy.py ..\\rabp_bmi_runs\\calibration_frontier\\calibration_frontier_logvar_intent_rest_t0.5-3.5\\summary.csv",
        ".\\.venv\\Scripts\\python scripts\\run_calibration_budget_frontier.py --config configs\\local.json --tmin 1.5 --tmax 4.0",
        ".\\.venv\\Scripts\\python scripts\\summarize_calibration_frontier.py ..\\rabp_bmi_runs\\calibration_frontier\\calibration_frontier_logvar_intent_rest_t1.5-4",
        ".\\.venv\\Scripts\\python scripts\\recommend_frontier_policy.py ..\\rabp_bmi_runs\\calibration_frontier\\calibration_frontier_logvar_intent_rest_t1.5-4\\summary.csv",
        ".\\.venv\\Scripts\\python scripts\\run_readiness_gate.py --config configs\\local.json --tmin 0.5 --tmax 3.5 --target-coverages 0.2,0.3,0.4,0.5,0.6 --output-suffix target_sweep",
        ".\\.venv\\Scripts\\python scripts\\summarize_readiness_gate.py ..\\rabp_bmi_runs\\readiness_gate\\readiness_gate_logvar_intent_rest_t0.5-3.5_target_sweep\\fold_results.csv",
        ".\\.venv\\Scripts\\python scripts\\select_readiness_policy.py ..\\rabp_bmi_runs\\readiness_gate\\readiness_gate_logvar_intent_rest_t0.5-3.5_target_sweep\\fold_results.csv",
        ".\\.venv\\Scripts\\python scripts\\run_meta_readiness_model.py ..\\rabp_bmi_runs\\readiness_gate\\readiness_gate_logvar_intent_rest_t0.5-3.5_target_sweep\\fold_results.csv",
        ".\\.venv\\Scripts\\python scripts\\summarize_meta_readiness.py ..\\rabp_bmi_runs\\readiness_gate\\readiness_gate_logvar_intent_rest_t0.5-3.5_target_sweep\\meta_readiness\\meta_predictions.csv",
        ".\\.venv\\Scripts\\python scripts\\run_readiness_gate.py --config configs\\local.json --tmin 1.5 --tmax 4.0",
        ".\\.venv\\Scripts\\python scripts\\summarize_readiness_gate.py ..\\rabp_bmi_runs\\readiness_gate\\readiness_gate_logvar_intent_rest_t1.5-4\\fold_results.csv",
        ".\\.venv\\Scripts\\python scripts\\select_readiness_policy.py ..\\rabp_bmi_runs\\readiness_gate\\readiness_gate_logvar_intent_rest_t1.5-4\\fold_results.csv",
        ".\\.venv\\Scripts\\python scripts\\run_meta_readiness_model.py ..\\rabp_bmi_runs\\readiness_gate\\readiness_gate_logvar_intent_rest_t1.5-4\\fold_results.csv",
        ".\\.venv\\Scripts\\python scripts\\summarize_meta_readiness.py ..\\rabp_bmi_runs\\readiness_gate\\readiness_gate_logvar_intent_rest_t1.5-4\\meta_readiness\\meta_predictions.csv",
        ".\\.venv\\Scripts\\python scripts\\statistical_validation.py --config configs\\local.json --n-bootstrap 5000",
        ".\\.venv\\Scripts\\python scripts\\analyze_channel_regions.py --config configs\\local.json",
        ".\\.venv\\Scripts\\python scripts\\make_policy_assets.py --config configs\\local.json",
        ".\\.venv\\Scripts\\python scripts\\build_robustness_summary.py --config configs\\local.json",
        ".\\.venv\\Scripts\\python scripts\\freeze_analysis_protocol.py --config configs\\local.json",
        ".\\.venv\\Scripts\\python scripts\\build_project_manifest.py --config configs\\local.json",
        ".\\.venv\\Scripts\\python scripts\\verify_core_outputs.py --config configs\\local.json --fail-on-error",
        ".\\.venv\\Scripts\\python scripts\\build_project_manifest.py --config configs\\local.json",
        ".\\.venv\\Scripts\\python scripts\\build_paper_results_package.py --config configs\\local.json --output-dir \"..\"",
        "",
        "# Heavy EEG-processing commands above can take time; manifest/assets commands are lightweight.",
    ]
    path.write_text("\n".join(commands) + "\n", encoding="utf-8")


def write_claim_register(path: Path) -> None:
    rows = [
        "# Claim Evidence Register\n",
        f"Date: {report_date()}\n",
        "| Claim ID | Status | Claim | Evidence |",
        "|---|---|---|---|",
    ]
    for item in CLAIMS:
        rows.append(
            f"| {item['claim_id']} | {item['status']} | "
            f"{item['claim']} | {item['evidence']} |"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_report(
    path: Path,
    *,
    manifest: dict[str, object],
    artifact_count: int,
    claim_count: int,
) -> None:
    dataset = manifest["dataset"]
    versions = manifest["environment"]["packages"]
    rows = [
        "# Milestone 09: Reproducibility Manifest\n",
        f"Date: {report_date()}\n",
        "## Purpose\n",
        "Create a stable audit trail for code, reports, results, command sequence, environment versions, and claim-to-evidence links.\n",
        "## Dataset State\n",
        "```text",
        f"target run files: {dataset.get('target_run_files')}",
        f"target subjects: {dataset.get('target_subjects')}",
        f"clean subjects: {dataset.get('clean_subjects')}",
        f"excluded subjects: {', '.join(dataset.get('excluded_subjects', []))}",
        "```",
        "## Environment\n",
        "```text",
        f"python: {manifest['environment']['python']}",
        f"platform: {manifest['environment']['platform']}",
    ]
    rows.extend(f"{package}: {version}" for package, version in versions.items())
    rows.extend(
        [
            "```",
            "## Manifest Outputs\n",
            "```text",
            "artifact_index.csv",
            "claim_evidence_register.csv",
            "claim_evidence_register.md",
            "command_sequence.ps1",
            "project_manifest.json",
            "reproducibility_report.md",
            "```",
            "## Summary\n",
            "```text",
            f"indexed artifacts: {artifact_count}",
            f"registered claims: {claim_count}",
            "```",
            "## Interpretation\n",
            "This manifest does not replace rerunning the full pipeline. It records the current validated state and links each major scientific claim to concrete evidence artifacts. Before manuscript submission, rerun the command sequence in a fresh folder or environment and compare the final manifest.",
        ]
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build reproducibility manifest and claim evidence register."
    )
    parser.add_argument("--config", default="configs/local.json")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(Path(args.config))
    out_dir = cfg.run_root / "project_manifest"
    out_dir.mkdir(parents=True, exist_ok=True)

    project_files = collect_project_files(project_root)
    result_files = collect_result_files(cfg.run_root)
    artifact_rows = [
        artifact_row(path, root=project_root, category="project")
        for path in project_files
    ]
    artifact_rows.extend(
        artifact_row(path, root=cfg.run_root, category="result")
        for path in result_files
    )
    artifact_rows = sorted(
        artifact_rows, key=lambda row: (str(row["category"]), str(row["relative_path"]))
    )
    write_csv(out_dir / "artifact_index.csv", artifact_rows)
    write_csv(out_dir / "claim_evidence_register.csv", CLAIMS)
    write_claim_register(out_dir / "claim_evidence_register.md")
    write_command_sequence(out_dir / "command_sequence.ps1")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "dataset_root": str(cfg.dataset_root),
        "run_root": str(cfg.run_root),
        "config": {
            "task_runs": list(cfg.task_runs),
            "random_seed": cfg.random_seed,
        },
        "dataset": dataset_summary(cfg.run_root / "audit" / "dataset_audit.csv"),
        "environment": {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "packages": package_versions(),
        },
        "artifact_index": "artifact_index.csv",
        "claim_evidence_register": "claim_evidence_register.csv",
        "command_sequence": "command_sequence.ps1",
        "artifact_count": len(artifact_rows),
        "claim_count": len(CLAIMS),
    }
    with (out_dir / "project_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    write_report(
        out_dir / "reproducibility_report.md",
        manifest=manifest,
        artifact_count=len(artifact_rows),
        claim_count=len(CLAIMS),
    )

    print(f"output_dir={out_dir}")
    print(f"artifact_count={len(artifact_rows)}")
    print(f"claim_count={len(CLAIMS)}")
    print(f"manifest={out_dir / 'project_manifest.json'}")
    print(f"artifact_index={out_dir / 'artifact_index.csv'}")
    print(f"claim_register={out_dir / 'claim_evidence_register.md'}")


if __name__ == "__main__":
    main()
