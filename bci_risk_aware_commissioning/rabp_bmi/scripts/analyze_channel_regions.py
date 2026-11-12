from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PROJECT_SRC))

from rabp_bmi.config import load_config
from rabp_bmi.features import MOTOR_CHANNELS


CALIBRATION_CONFIGS = (
    ("topk_effect", 16, 15),
    ("topk_effect", 21, 15),
    ("topk_effect", 21, 12),
    ("central_parietal_topk_effect", 21, 15),
    ("fixed_motor", 21, 15),
    ("all", 64, 15),
)

BUDGETED_CONFIGS = (
    ("topk_effect", 16, None),
    ("topk_effect", 21, None),
    ("central_parietal_topk_effect", 21, None),
    ("motor_topk_effect", 16, None),
    ("motor_topk_effect", 21, None),
    ("fixed_motor", 21, None),
    ("all", 64, None),
)


def report_date() -> str:
    return datetime.now().date().isoformat()


def channel_region(channel: str) -> str:
    name = channel.upper()
    if name in {"IZ", "OZ"} or name.startswith("O") or name.startswith("PO"):
        return "posterior_occipital"
    if name.startswith("FT") or name.startswith("TP") or name.startswith("T"):
        return "temporal"
    if name.startswith("FC"):
        return "frontocentral"
    if name.startswith("CP"):
        return "centroparietal"
    if name.startswith("C"):
        return "central"
    if name.startswith("FP") or name.startswith("AF") or name.startswith("F"):
        return "frontal"
    if name.startswith("P"):
        return "parietal"
    return "other"


def split_channels(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip().upper() for part in str(value).split() if part.strip()]


def load_selected_configs(
    path: Path,
    *,
    source: str,
    configs: tuple[tuple[str, int, int | None], ...],
) -> pd.DataFrame:
    selected = pd.read_csv(path)
    rows: list[dict[str, object]] = []
    for policy, budget, m_value in configs:
        mask = (selected["policy"] == policy) & (selected["budget"] == budget)
        if m_value is not None:
            mask = mask & (selected["m_per_class_per_run"] == m_value)
        subset = selected[mask].copy()
        if subset.empty:
            continue
        for _, row in subset.iterrows():
            fold_id = f"{row['subject']}_{int(row['outer_test_run'])}"
            channels = split_channels(row["selected_channels"])
            for channel in channels:
                rows.append(
                    {
                        "source": source,
                        "subject": row["subject"],
                        "outer_test_run": int(row["outer_test_run"]),
                        "fold_id": fold_id,
                        "policy": policy,
                        "budget": int(budget),
                        "m_per_class_per_run": int(m_value)
                        if m_value is not None
                        else -1,
                        "channel": channel,
                        "region": channel_region(channel),
                        "in_motor_set": channel in MOTOR_CHANNELS,
                    }
                )
    if not rows:
        raise ValueError(f"No selected-channel rows found in {path}")
    return pd.DataFrame(rows)


def channel_frequency(long_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["source", "policy", "budget", "m_per_class_per_run"]
    n_folds = (
        long_df[group_cols + ["fold_id"]]
        .drop_duplicates()
        .groupby(group_cols)
        .size()
        .rename("n_folds")
        .reset_index()
    )
    counts = (
        long_df.groupby(group_cols + ["channel", "region", "in_motor_set"])
        .size()
        .rename("selection_count")
        .reset_index()
        .merge(n_folds, on=group_cols, how="left")
    )
    counts["selection_rate"] = counts["selection_count"] / counts["n_folds"]
    counts = counts.sort_values(
        group_cols + ["selection_count", "channel"],
        ascending=[True, True, True, True, False, True],
    )
    counts["rank"] = counts.groupby(group_cols)["selection_count"].rank(
        method="first", ascending=False
    )
    return counts


def region_composition(long_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["source", "policy", "budget", "m_per_class_per_run"]
    counts = (
        long_df.groupby(group_cols + ["region"])
        .size()
        .rename("selected_channel_instances")
        .reset_index()
    )
    totals = (
        long_df.groupby(group_cols)
        .size()
        .rename("total_selected_channel_instances")
        .reset_index()
    )
    counts = counts.merge(totals, on=group_cols, how="left")
    counts["region_proportion"] = (
        counts["selected_channel_instances"]
        / counts["total_selected_channel_instances"]
    )
    return counts.sort_values(group_cols + ["region"])


def policy_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["source", "policy", "budget", "m_per_class_per_run"]
    per_fold = (
        long_df.groupby(group_cols + ["fold_id"])
        .agg(
            n_selected=("channel", "size"),
            n_motor_set=("in_motor_set", "sum"),
            n_posterior=(
                "region",
                lambda s: int((s == "posterior_occipital").sum()),
            ),
            n_frontal=("region", lambda s: int((s == "frontal").sum())),
        )
        .reset_index()
    )
    summary = (
        per_fold.groupby(group_cols)
        .agg(
            n_folds=("fold_id", "size"),
            mean_selected=("n_selected", "mean"),
            mean_motor_set_channels=("n_motor_set", "mean"),
            mean_motor_set_fraction=(
                "n_motor_set",
                lambda s: float(s.sum() / per_fold.loc[s.index, "n_selected"].sum()),
            ),
            mean_posterior_channels=("n_posterior", "mean"),
            mean_posterior_fraction=(
                "n_posterior",
                lambda s: float(s.sum() / per_fold.loc[s.index, "n_selected"].sum()),
            ),
            mean_frontal_channels=("n_frontal", "mean"),
            mean_frontal_fraction=(
                "n_frontal",
                lambda s: float(s.sum() / per_fold.loc[s.index, "n_selected"].sum()),
            ),
        )
        .reset_index()
    )
    return summary


def write_report(out_path: Path, summary: pd.DataFrame, freq: pd.DataFrame) -> None:
    rows: list[str] = []
    rows.append("# Channel-Region Audit\n")
    rows.append(f"Date: {report_date()}\n")
    rows.append("## Purpose\n")
    rows.append(
        "Quantify whether selected-channel policies rely mainly on canonical "
        "motor electrodes or on broader frontal/posterior electrodes. This is "
        "important because the task is cue-paced command gating, not a pure "
        "self-paced motor-intention decoder.\n"
    )
    rows.append("## Policy Summary\n")
    rows.append(
        "| Source | Policy | Budget | m | Motor Fraction | Posterior Fraction | Frontal Fraction |"
    )
    rows.append("|---|---|---:|---:|---:|---:|---:|")
    for _, row in summary.iterrows():
        m_value = int(row["m_per_class_per_run"])
        m_text = "full" if m_value < 0 else str(m_value)
        rows.append(
            "| "
            f"{row['source']} | {row['policy']} | {int(row['budget'])} | {m_text} | "
            f"{row['mean_motor_set_fraction']:.3f} | "
            f"{row['mean_posterior_fraction']:.3f} | "
            f"{row['mean_frontal_fraction']:.3f} |"
        )
    rows.append("\n## Top Channels\n")
    rows.append("| Source | Policy | Budget | m | Top channels |")
    rows.append("|---|---|---:|---:|---|")
    for keys, group in freq.groupby(
        ["source", "policy", "budget", "m_per_class_per_run"], sort=False
    ):
        source, policy, budget, m_value = keys
        m_text = "full" if int(m_value) < 0 else str(int(m_value))
        top = group.sort_values("rank").head(10)
        top_text = ", ".join(
            f"{row.channel} ({row.selection_rate:.2f})"
            for row in top.itertuples(index=False)
        )
        rows.append(f"| {source} | {policy} | {int(budget)} | {m_text} | {top_text} |")
    rows.append("\n## Interpretation\n")
    rows.append(
        "If an unconstrained top-k policy selects many posterior or frontal "
        "channels, frame it as a cue-paced command-gating policy. Do not present "
        "it as evidence of pure sensorimotor-rhythm decoding. Central-parietal "
        "and motor-restricted variants remain important biological sensitivity "
        "checks.\n"
    )
    out_path.write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit selected EEG channel regions.")
    parser.add_argument("--config", default="configs/local.json")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    calibration_path = (
        cfg.run_root
        / "calibration_frontier"
        / "calibration_frontier_logvar_intent_rest_t0.5-3.5"
        / "selected_channels.csv"
    )
    budgeted_path = (
        cfg.run_root
        / "budgeted_rejection"
        / "budgeted_logvar_intent_rest_policies_v2_t0.5-3.5"
        / "selected_channels.csv"
    )
    calibration_long = load_selected_configs(
        calibration_path, source="calibration_limited", configs=CALIBRATION_CONFIGS
    )
    budgeted_long = load_selected_configs(
        budgeted_path, source="full_training", configs=BUDGETED_CONFIGS
    )
    long_df = pd.concat([calibration_long, budgeted_long], ignore_index=True)

    freq = channel_frequency(long_df)
    regions = region_composition(long_df)
    summary = policy_summary(long_df)

    out_dir = cfg.run_root / "validation" / "channel_region_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(out_dir / "selected_channel_long.csv", index=False)
    freq.to_csv(out_dir / "channel_frequency.csv", index=False)
    regions.to_csv(out_dir / "region_composition.csv", index=False)
    summary.to_csv(out_dir / "policy_region_summary.csv", index=False)
    write_report(out_dir / "channel_region_audit.md", summary, freq)

    print(f"output_dir={out_dir}")
    print(f"policy_region_summary={out_dir / 'policy_region_summary.csv'}")
    print(f"channel_frequency={out_dir / 'channel_frequency.csv'}")
    print(
        summary[
            [
                "source",
                "policy",
                "budget",
                "m_per_class_per_run",
                "mean_motor_set_fraction",
                "mean_posterior_fraction",
                "mean_frontal_fraction",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )


if __name__ == "__main__":
    main()
