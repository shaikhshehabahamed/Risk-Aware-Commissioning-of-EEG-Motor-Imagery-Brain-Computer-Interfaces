from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PROJECT_SRC))

from rabp_bmi.config import load_config


PACKAGE_VERSION = "v2.0"
MANAGED_PACKAGE_FILES = (
    "PAPER_RESULTS_INDEX.md",
    "figures/figure_1_policy_tradeoff_unconditional_fcr_vs_ba.pdf",
    "figures/figure_1_policy_tradeoff_unconditional_fcr_vs_ba.png",
    "figures/figure_2_operating_point_dashboard.pdf",
    "figures/figure_2_operating_point_dashboard.png",
    "figures/figure_3_channel_region_composition.pdf",
    "figures/figure_3_channel_region_composition.png",
    "figures/figure_s1_window_sensitivity.pdf",
    "figures/figure_s1_window_sensitivity.png",
    "tables/table_1_operating_points.csv",
    "tables/table_2_key_statistical_comparisons.csv",
    "tables/table_3_channel_region_sensitivity.csv",
    "tables/table_s1_window_sensitivity.csv",
)

SHORT_LABELS = {
    "topk21_m15_rule_t03_ba065_fcr010": "Primary21",
    "topk16_m15_rule_t03_ba070_fcr010": "Minimal16",
    "all64_m15_no_gate": "All64",
    "topk16_m15_no_gate": "TopK16",
    "topk21_m15_no_gate": "TopK21",
    "topk16_m15_rule_ba075_fcr010": "Rule75",
    "topk21_m15_t03_meta_p060": "Meta21",
    "central21_m15_no_gate": "Central21",
    "fixed_motor21_m15_no_gate": "Motor21",
}

POLICY_LABELS = {
    ("all", 64): "All64",
    ("topk_effect", 16): "TopK16",
    ("topk_effect", 21): "TopK21",
    ("central_parietal_topk_effect", 21): "Central21",
    ("fixed_motor", 21): "Motor21",
}

KEY_COMPARISON_FILTERS = [
    ("topk16_m15_no_gate", "all64_m15_no_gate", "accepted_balanced_accuracy"),
    ("topk16_m15_no_gate", "all64_m15_no_gate", "unconditional_false_command_rate"),
    ("topk16_m15_no_gate", "all64_m15_no_gate", "unconditional_actual_coverage"),
    ("topk21_m15_no_gate", "all64_m15_no_gate", "accepted_balanced_accuracy"),
    ("topk21_m15_no_gate", "all64_m15_no_gate", "unconditional_false_command_rate"),
    ("topk21_m15_rule_t03_ba065_fcr010", "all64_m15_no_gate", "unconditional_false_command_rate"),
    ("topk21_m15_rule_t03_ba065_fcr010", "all64_m15_no_gate", "unconditional_actual_coverage"),
    ("topk21_m15_rule_t03_ba065_fcr010", "all64_m15_no_gate", "deployment_rate"),
    ("topk21_m15_rule_t03_ba065_fcr010", "topk21_m15_no_gate", "unconditional_false_command_rate"),
    ("topk21_m15_rule_t03_ba065_fcr010", "topk21_m15_no_gate", "unconditional_actual_coverage"),
    ("topk16_m15_rule_t03_ba070_fcr010", "all64_m15_no_gate", "deployment_rate"),
    ("topk21_m15_t03_meta_p060", "topk21_m15_no_gate", "unconditional_false_command_rate"),
    ("topk21_m15_t03_meta_p060", "topk21_m15_no_gate", "deployment_rate"),
]


def report_date() -> str:
    return datetime.now().date().isoformat()


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_ci(value: str) -> tuple[float, float, float]:
    point_text, interval_text = value.split("[", maxsplit=1)
    low_text, high_text = interval_text.rstrip("]").split(",", maxsplit=1)
    return float(point_text.strip()), float(low_text.strip()), float(high_text.strip())


def build_operating_points(hierarchy: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "rank",
        "mode_label",
        "role",
        "display_name",
        "mean_channels",
        "mean_calibration_trials",
        "deployment_rate_ci",
        "deployed_coverage_ci",
        "accepted_balanced_accuracy_ci",
        "deployed_false_command_rate_ci",
        "unconditional_false_command_rate_ci",
        "unconditional_coverage_ci",
    ]
    return hierarchy[columns].copy()


def build_key_comparisons(comparisons: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for challenger, comparator, metric in KEY_COMPARISON_FILTERS:
        subset = comparisons[
            (comparisons["challenger"] == challenger)
            & (comparisons["comparator"] == comparator)
            & (comparisons["metric"] == metric)
        ]
        if subset.empty:
            raise ValueError(
                "Missing key comparison: "
                f"{challenger} vs {comparator} for {metric}"
            )
        rows.append(subset.iloc[0])
    out = pd.DataFrame(rows).copy()
    return out[
        [
            "challenger",
            "comparator",
            "metric",
            "n_pairs",
            "mean_challenger",
            "mean_comparator",
            "mean_difference",
            "ci_low",
            "ci_high",
            "wilcoxon_p_holm_by_metric",
        ]
    ]


def build_channel_region_table(regions: pd.DataFrame) -> pd.DataFrame:
    keep = (
        (regions["source"] == "calibration_limited")
        & (regions["m_per_class_per_run"] == 15)
        & (
            regions["policy"].isin(
                ["all", "topk_effect", "central_parietal_topk_effect", "fixed_motor"]
            )
        )
        & (regions["budget"].isin([16, 21, 64]))
    )
    out = regions[keep].copy()
    return out[
        [
            "policy",
            "budget",
            "mean_selected",
            "mean_motor_set_fraction",
            "mean_posterior_fraction",
            "mean_frontal_fraction",
        ]
    ].sort_values(["policy", "budget"])


def plot_operating_dashboard(hierarchy: pd.DataFrame, output_dir: Path) -> None:
    plot_df = hierarchy.copy().sort_values("rank", ascending=False)
    labels = [SHORT_LABELS.get(mode, mode) for mode in plot_df["mode_label"]]
    colors = [
        "#1f77b4" if role == "primary_policy" else "#666666"
        for role in plot_df["role"]
    ]
    metrics = [
        ("deployment_rate_ci", "Deployment", (0.0, 1.05)),
        ("accepted_balanced_accuracy_ci", "Accepted BA", (0.70, 0.98)),
        ("unconditional_false_command_rate_ci", "Unconditional FCR", (0.0, 0.08)),
        ("unconditional_coverage_ci", "Unconditional coverage", (0.0, 0.36)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    y = range(len(plot_df))
    for ax, (column, title, xlim) in zip(axes.ravel(), metrics, strict=True):
        parsed = [parse_ci(value) for value in plot_df[column]]
        points = [item[0] for item in parsed]
        lows = [item[1] for item in parsed]
        highs = [item[2] for item in parsed]
        xerr = [
            [point - low for point, low in zip(points, lows, strict=True)],
            [high - point for point, high in zip(points, highs, strict=True)],
        ]
        ax.errorbar(
            points,
            y,
            xerr=xerr,
            fmt="o",
            markersize=5,
            color="#222222",
            ecolor="#777777",
            elinewidth=1,
            capsize=3,
        )
        ax.scatter(points, y, s=42, c=colors, zorder=3)
        ax.set_title(title)
        ax.set_xlim(*xlim)
        ax.grid(axis="x", alpha=0.25)
        ax.set_yticks(list(y), labels)
    fig.suptitle("Operating-point performance with subject-level 95% CIs", y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir, "figure_2_operating_point_dashboard")


def plot_channel_regions(channel_regions: pd.DataFrame, output_dir: Path) -> None:
    plot_df = channel_regions.copy()
    plot_df["label"] = [
        POLICY_LABELS.get((row.policy, int(row.budget)), f"{row.policy}{int(row.budget)}")
        for row in plot_df.itertuples(index=False)
    ]
    order = ["All64", "TopK16", "TopK21", "Central21", "Motor21"]
    plot_df["order"] = plot_df["label"].map({label: i for i, label in enumerate(order)})
    plot_df = plot_df.sort_values("order")
    plot_df["other_fraction"] = (
        1.0
        - plot_df["mean_motor_set_fraction"]
        - plot_df["mean_posterior_fraction"]
        - plot_df["mean_frontal_fraction"]
    ).clip(lower=0.0)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    left = [0.0] * len(plot_df)
    stacks = [
        ("mean_motor_set_fraction", "Motor set", "#4c78a8"),
        ("mean_posterior_fraction", "Posterior", "#f58518"),
        ("mean_frontal_fraction", "Frontal", "#54a24b"),
        ("other_fraction", "Other", "#b0b0b0"),
    ]
    y = range(len(plot_df))
    for column, label, color in stacks:
        values = list(plot_df[column])
        ax.barh(y, values, left=left, label=label, color=color)
        left = [base + value for base, value in zip(left, values, strict=True)]
    ax.set_yticks(list(y), list(plot_df["label"]))
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of selected channels")
    ax.set_title("Channel-region composition of selected policies")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.26), ncol=4, frameon=False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_dir, "figure_3_channel_region_composition")


def plot_window_sensitivity(window: pd.DataFrame, output_dir: Path) -> None:
    plot_df = window.copy()
    label_map = {
        ("all", 64): "All64",
        ("topk_effect", 16): "TopK16",
        ("topk_effect", 21): "TopK21",
        ("topk_effect", 32): "TopK32",
        ("fixed_motor", 21): "Motor21",
        ("motor_topk_effect", 16): "MotorTopK16",
        ("motor_topk_effect", 21): "MotorTopK21",
    }
    plot_df["label"] = [
        label_map.get((row.policy, int(row.budget)), f"{row.policy}{int(row.budget)}")
        for row in plot_df.itertuples(index=False)
    ]
    plot_df = plot_df.iloc[::-1]
    y = range(len(plot_df))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    specs = [
        ("original_accepted_ba", "delayed_accepted_ba", "Accepted BA", (0.70, 0.88)),
        ("original_fcr", "delayed_fcr", "False-command rate", (0.035, 0.075)),
    ]
    for ax, (early_col, delayed_col, title, xlim) in zip(axes, specs, strict=True):
        early = list(plot_df[early_col])
        delayed = list(plot_df[delayed_col])
        for i, (early_value, delayed_value) in enumerate(zip(early, delayed, strict=True)):
            ax.plot([early_value, delayed_value], [i, i], color="#777777", linewidth=1.5)
        ax.scatter(early, y, label="0.5-3.5 s", color="#1f77b4", zorder=3)
        ax.scatter(delayed, y, label="1.5-4.0 s", color="#d62728", zorder=3)
        ax.set_title(title)
        ax.set_xlim(*xlim)
        ax.grid(axis="x", alpha=0.25)
        ax.set_yticks(list(y), list(plot_df["label"]))
    axes[0].legend(loc="lower left", frameon=False)
    fig.suptitle("Primary-window versus delayed-window sensitivity", y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir, "figure_s1_window_sensitivity")


def write_index(
    path: Path,
    *,
    operating_points: pd.DataFrame,
    key_comparisons: pd.DataFrame,
) -> None:
    primary = operating_points[operating_points["role"] == "primary_policy"].iloc[0]
    rows = [
        f"# Paper-Ready Results {PACKAGE_VERSION}",
        "",
        f"Date: {report_date()}",
        "",
        "## Scope",
        "",
        "This index covers only manuscript-facing result files: final CSV tables",
        "and focused result figures in publication and preview formats.",
        "Reproducibility manifests, protocol files, and QA logs remain in the",
        "working run directory, not among these paper-ready files.",
        "",
        "## Primary Result",
        "",
        "```text",
        f"policy: {primary['mode_label']}",
        "window: 0.5-3.5 s post-cue",
        "channels: 21",
        "calibration trials: 60",
        "target coverage: 0.3",
        "readiness rule: calibration BA >= 0.65 and calibration FCR <= 0.10",
        f"deployment: {primary['deployment_rate_ci']}",
        f"accepted BA: {primary['accepted_balanced_accuracy_ci']}",
        f"deployed FCR: {primary['deployed_false_command_rate_ci']}",
        f"unconditional FCR: {primary['unconditional_false_command_rate_ci']}",
        f"unconditional coverage: {primary['unconditional_coverage_ci']}",
        "```",
        "",
        "## Files",
        "",
        "| File | Format | Paper use |",
        "|---|---|---|",
        "| figures/figure_1_policy_tradeoff_unconditional_fcr_vs_ba.pdf/png | PDF, PNG | Main safety-reliability tradeoff figure |",
        "| figures/figure_2_operating_point_dashboard.pdf/png | PDF, PNG | Main operating-point summary figure |",
        "| figures/figure_3_channel_region_composition.pdf/png | PDF, PNG | Channel-region interpretation figure |",
        "| figures/figure_s1_window_sensitivity.pdf/png | PDF, PNG | Supplementary robustness figure |",
        "| tables/table_1_operating_points.csv | CSV | Main operating-point table |",
        "| tables/table_2_key_statistical_comparisons.csv | CSV | Main or supplementary paired-comparison table |",
        "| tables/table_3_channel_region_sensitivity.csv | CSV | Channel-region sensitivity table |",
        "| tables/table_s1_window_sensitivity.csv | CSV | Supplementary window-sensitivity table |",
        "",
        "## Guardrail",
        "",
        "Frame the contribution as risk-aware biomedical decision analytics for",
        "cue-paced EEG command gating. Do not claim pure motor-cortex decoding,",
        "self-paced asynchronous BMI, clinical effectiveness, universal deployment,",
        "or new-classifier superiority.",
        "",
        "## Export Metadata",
        "",
        "```text",
        f"created_utc: {datetime.now(timezone.utc).isoformat()}",
        f"key_comparison_rows: {len(key_comparisons)}",
        "```",
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def build_package(run_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in MANAGED_PACKAGE_FILES:
        path = output_dir / item
        if path.is_dir():
            raise IsADirectoryError(f"Refusing to replace directory as managed file: {path}")
        if path.exists():
            path.unlink()

    protocol_dir = run_root / "final_protocol" / "analysis_protocol_v2.0"
    validation_dir = run_root / "validation"
    assets_dir = run_root / "paper_assets" / "operating_points"

    hierarchy = pd.read_csv(protocol_dir / "operating_point_hierarchy.csv")
    comparisons = pd.read_csv(
        validation_dir / "subject_level_policy_validation" / "paired_comparisons.csv"
    )
    regions = pd.read_csv(validation_dir / "channel_region_audit" / "policy_region_summary.csv")
    window = pd.read_csv(validation_dir / "robustness_summary" / "window_sensitivity.csv")

    operating_points = build_operating_points(hierarchy)
    key_comparisons = build_key_comparisons(comparisons)
    channel_regions = build_channel_region_table(regions)

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    operating_points.to_csv(tables_dir / "table_1_operating_points.csv", index=False)
    key_comparisons.to_csv(tables_dir / "table_2_key_statistical_comparisons.csv", index=False)
    channel_regions.to_csv(tables_dir / "table_3_channel_region_sensitivity.csv", index=False)
    window.to_csv(tables_dir / "table_s1_window_sensitivity.csv", index=False)

    copy_file(
        assets_dir / "policy_tradeoff_unconditional_fcr_vs_ba.pdf",
        output_dir / "figures" / "figure_1_policy_tradeoff_unconditional_fcr_vs_ba.pdf",
    )
    copy_file(
        assets_dir / "policy_tradeoff_unconditional_fcr_vs_ba.png",
        output_dir / "figures" / "figure_1_policy_tradeoff_unconditional_fcr_vs_ba.png",
    )
    plot_operating_dashboard(hierarchy, output_dir / "figures")
    plot_channel_regions(channel_regions, output_dir / "figures")
    plot_window_sensitivity(window, output_dir / "figures")

    write_index(
        output_dir / "PAPER_RESULTS_INDEX.md",
        operating_points=operating_points,
        key_comparisons=key_comparisons,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a minimal paper-ready results package."
    )
    parser.add_argument("--config", default="configs/local.json")
    parser.add_argument(
        "--output-dir",
        default="..",
        help="Destination folder for the managed paper-ready results files.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(Path(args.config))
    output_dir_arg = Path(args.output_dir).expanduser()
    if output_dir_arg.is_absolute():
        output_dir = output_dir_arg.resolve()
    else:
        output_dir = (project_root / output_dir_arg).resolve()
    build_package(cfg.run_root, output_dir)
    print(f"output_dir={output_dir}")
    print(f"index={output_dir / 'PAPER_RESULTS_INDEX.md'}")


if __name__ == "__main__":
    main()
