from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PROJECT_SRC))

from rabp_bmi.config import load_config


DISPLAY_ORDER = [
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

SHORT_LABELS = {
    "all64_m15_no_gate": "All64",
    "topk16_m15_no_gate": "TopK16",
    "topk21_m15_no_gate": "TopK21",
    "central21_m15_no_gate": "Central21",
    "fixed_motor21_m15_no_gate": "Motor21",
    "topk16_m15_rule_ba075_fcr010": "Rule75",
}

LABEL_OFFSETS = {
    "All64": (7, -1),
    "TopK16": (7, 8),
    "TopK21": (7, -10),
    "Central21": (7, 8),
    "Motor21": (7, -2),
    "Primary21": (7, 8),
    "Minimal16": (7, -10),
    "Rule75": (7, 6),
}

SELECTED_FOR_TABLE = [
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

SHORT_LABELS.update(
    {
        "topk21_m15_rule_t03_ba065_fcr010": "Primary21",
        "topk16_m15_rule_t03_ba070_fcr010": "Minimal16",
        "topk21_m15_t03_meta_p060": "Meta21",
    }
)

METRICS_FOR_TABLE = [
    "deployment_rate",
    "deployed_actual_coverage",
    "accepted_balanced_accuracy",
    "false_command_rate_per_rest_opportunity",
    "unconditional_actual_coverage",
    "unconditional_false_command_rate",
]


def report_date() -> str:
    return datetime.now().date().isoformat()


def metric_ci(ci_df: pd.DataFrame, mode_label: str, metric: str) -> tuple[float, float, float]:
    row = ci_df[(ci_df["mode_label"] == mode_label) & (ci_df["metric"] == metric)]
    if row.empty:
        raise ValueError(f"Missing metric {metric} for {mode_label}")
    item = row.iloc[0]
    return float(item["point_estimate"]), float(item["ci_low"]), float(item["ci_high"])


def fmt_ci(point: float, low: float, high: float) -> str:
    return f"{point:.3f} [{low:.3f}, {high:.3f}]"


def build_operating_table(ci_df: pd.DataFrame, aggregate_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    aggregate = aggregate_df.set_index("mode_label")
    for mode in SELECTED_FOR_TABLE:
        agg = aggregate.loc[mode]
        row: dict[str, object] = {
            "mode_label": mode,
            "label": SHORT_LABELS[mode],
            "policy": agg["display_name"],
            "mean_channels": float(agg["mean_deployed_channel_budget"]),
            "mean_calibration_trials": float(agg["mean_deployed_calibration_trials"]),
        }
        for metric in METRICS_FOR_TABLE:
            point, low, high = metric_ci(ci_df, mode, metric)
            row[metric] = point
            row[f"{metric}_ci"] = fmt_ci(point, low, high)
        rows.append(row)
    return pd.DataFrame(rows)


def write_markdown_table(path: Path, table: pd.DataFrame) -> None:
    rows = [
        "# Operating-Point Table\n",
        f"Date: {report_date()}\n",
        "| Label | Mean Ch. | Calib. Trials | Deploy | Deployed Cov. | Accepted BA | Deployed FCR | Uncond. Cov. | Uncond. FCR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table.itertuples(index=False):
        rows.append(
            "| "
            f"{row.label} | "
            f"{row.mean_channels:.1f} | "
            f"{row.mean_calibration_trials:.1f} | "
            f"{row.deployment_rate_ci} | "
            f"{row.deployed_actual_coverage_ci} | "
            f"{row.accepted_balanced_accuracy_ci} | "
            f"{row.false_command_rate_per_rest_opportunity_ci} | "
            f"{row.unconditional_actual_coverage_ci} | "
            f"{row.unconditional_false_command_rate_ci} |"
        )
    rows.append("\nNotes:")
    rows.append("- Intervals are 95% subject-level bootstrap confidence intervals.")
    rows.append("- Deployed metrics condition on sessions accepted by the policy.")
    rows.append("- Unconditional metrics include no-go sessions in the denominator.")
    path.write_text("\n".join(rows), encoding="utf-8")


def make_tradeoff_plot(ci_df: pd.DataFrame, aggregate_df: pd.DataFrame, out_dir: Path) -> None:
    aggregate = aggregate_df.set_index("mode_label")
    records = []
    for mode in DISPLAY_ORDER:
        if mode not in aggregate.index:
            continue
        ba = metric_ci(ci_df, mode, "accepted_balanced_accuracy")
        fcr = metric_ci(ci_df, mode, "unconditional_false_command_rate")
        coverage = metric_ci(ci_df, mode, "unconditional_actual_coverage")
        deploy = metric_ci(ci_df, mode, "deployment_rate")
        records.append(
            {
                "mode_label": mode,
                "label": SHORT_LABELS[mode],
                "ba": ba[0],
                "ba_low": ba[1],
                "ba_high": ba[2],
                "fcr": fcr[0],
                "fcr_low": fcr[1],
                "fcr_high": fcr[2],
                "coverage": coverage[0],
                "deployment": deploy[0],
            }
        )
    df = pd.DataFrame(records)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    scatter = ax.scatter(
        df["fcr"],
        df["ba"],
        s=80 + 260 * df["deployment"],
        c=df["coverage"],
        cmap="viridis",
        edgecolor="black",
        linewidth=0.6,
        alpha=0.92,
    )
    for row in df.itertuples(index=False):
        ax.errorbar(
            row.fcr,
            row.ba,
            xerr=[[row.fcr - row.fcr_low], [row.fcr_high - row.fcr]],
            yerr=[[row.ba - row.ba_low], [row.ba_high - row.ba]],
            fmt="none",
            ecolor="0.55",
            elinewidth=0.8,
            capsize=2,
            alpha=0.7,
        )
        ax.annotate(
            row.label,
            (row.fcr, row.ba),
            xytext=LABEL_OFFSETS.get(row.label, (5, 4)),
            textcoords="offset points",
            fontsize=8.5,
        )
    ax.set_xlabel("Unconditional false-command rate")
    ax.set_ylabel("Accepted balanced accuracy")
    ax.set_title("Risk-reliability operating points")
    ax.set_xlim(max(0.0, df["fcr_low"].min() - 0.004), df["fcr_high"].max() + 0.004)
    ax.set_ylim(df["ba_low"].min() - 0.012, df["ba_high"].max() + 0.009)
    ax.grid(True, color="0.9", linewidth=0.7)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Unconditional command coverage")
    fig.tight_layout()
    fig.savefig(out_dir / "policy_tradeoff_unconditional_fcr_vs_ba.png", dpi=300)
    fig.savefig(out_dir / "policy_tradeoff_unconditional_fcr_vs_ba.pdf")
    plt.close(fig)


def write_recommendation_note(path: Path, table: pd.DataFrame) -> None:
    rows = [
        "# Operating-Point Recommendation\n",
        f"Date: {report_date()}\n",
        "## Primary Policy\n",
        "Use `Primary21` as the final v2 reliability-safety operating point:",
        "",
        "```text",
        "0.5-3.5 s post-cue window",
        "top-k 21 channels, 60 calibration trials",
        "target coverage 0.3",
        "deploy if calibration accepted BA >= 0.65 and calibration FCR <= 0.10",
        "```",
        "",
        "Rationale:",
        "",
        "- interpretable readiness/no-go rule;",
        "- 21 channels instead of 64;",
        "- accepted BA around 0.858;",
        "- unconditional false-command rate around 0.016;",
        "- deployment rate around 0.571 with unconditional command coverage around 0.146.",
        "",
        "## Secondary Policies",
        "",
        "`Minimal16` is the main lower-channel secondary policy. It preserves the 16-channel burden while improving deployment relative to the conservative delayed-window policy.",
        "",
        "`TopK16` and `TopK21` no-gate policies are always-deploy budgeted baselines.",
        "",
        "`Meta21` is a high-confidence sensitivity mode, not the central contribution.",
        "",
        "`Central21` and `Motor21` remain biological plausibility sensitivity policies.",
        "",
        "## Table Reference",
        "",
    ]
    for row in table.itertuples(index=False):
        rows.append(
            f"- {row.label}: BA {row.accepted_balanced_accuracy:.3f}, "
            f"unconditional FCR {row.unconditional_false_command_rate:.3f}, "
            f"deployment {row.deployment_rate:.3f}."
        )
    path.write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create paper-facing operating-point tables and figures."
    )
    parser.add_argument("--config", default="configs/local.json")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    validation_dir = cfg.run_root / "validation" / "subject_level_policy_validation"
    ci_df = pd.read_csv(validation_dir / "bootstrap_confidence_intervals.csv")
    aggregate_df = pd.read_csv(validation_dir / "aggregate_summary.csv")

    out_dir = cfg.run_root / "paper_assets" / "operating_points"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = build_operating_table(ci_df, aggregate_df)
    table.to_csv(out_dir / "operating_point_table.csv", index=False)
    write_markdown_table(out_dir / "operating_point_table.md", table)
    write_recommendation_note(out_dir / "operating_point_recommendation.md", table)
    make_tradeoff_plot(ci_df, aggregate_df, out_dir)

    print(f"output_dir={out_dir}")
    print(f"table={out_dir / 'operating_point_table.csv'}")
    print(f"figure={out_dir / 'policy_tradeoff_unconditional_fcr_vs_ba.png'}")
    print(
        table[
            [
                "label",
                "mean_channels",
                "mean_calibration_trials",
                "deployment_rate",
                "accepted_balanced_accuracy",
                "unconditional_false_command_rate",
                "unconditional_actual_coverage",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )


if __name__ == "__main__":
    main()
