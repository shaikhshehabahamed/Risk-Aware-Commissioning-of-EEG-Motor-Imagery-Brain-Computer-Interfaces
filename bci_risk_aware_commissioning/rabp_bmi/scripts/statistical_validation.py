from __future__ import annotations

import argparse
from datetime import datetime
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

try:
    from scipy.stats import binomtest
except ImportError:  # pragma: no cover - for older SciPy only.
    from scipy.stats import binom_test

    def binomtest(k: int, n: int, p: float = 0.5, alternative: str = "two-sided"):
        class Result:
            def __init__(self, pvalue: float) -> None:
                self.pvalue = pvalue

        return Result(float(binom_test(k, n, p=p, alternative=alternative)))


PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PROJECT_SRC))

from rabp_bmi.config import load_config


NO_GATE_SPECS = (
    {
        "mode_label": "all64_m15_no_gate",
        "display_name": "All channels, 60 calibration trials",
        "policy": "all",
        "budget": 64,
        "m": 15,
        "target": 0.4,
    },
    {
        "mode_label": "fixed_motor21_m15_no_gate",
        "display_name": "Fixed motor montage, 21 channels, 60 trials",
        "policy": "fixed_motor",
        "budget": 21,
        "m": 15,
        "target": 0.4,
    },
    {
        "mode_label": "topk16_m15_no_gate",
        "display_name": "Top-k selected, 16 channels, 60 trials",
        "policy": "topk_effect",
        "budget": 16,
        "m": 15,
        "target": 0.4,
    },
    {
        "mode_label": "topk21_m12_no_gate",
        "display_name": "Top-k selected, 21 channels, 48 trials",
        "policy": "topk_effect",
        "budget": 21,
        "m": 12,
        "target": 0.4,
    },
    {
        "mode_label": "topk21_m15_no_gate",
        "display_name": "Top-k selected, 21 channels, 60 trials",
        "policy": "topk_effect",
        "budget": 21,
        "m": 15,
        "target": 0.4,
    },
    {
        "mode_label": "central21_m15_no_gate",
        "display_name": "Central-parietal selected, 21 channels, 60 trials",
        "policy": "central_parietal_topk_effect",
        "budget": 21,
        "m": 15,
        "target": 0.4,
    },
)

RULE_GATE_SPECS = (
    {
        "mode_label": "topk21_m15_rule_t03_ba065_fcr010",
        "display_name": "Top-k 21, target 0.3, rule gate: calib BA >= 0.65 and FCR <= 0.10",
        "candidate": "topk_effect_k21_m15_target0.3",
        "gate_min_ba": 0.65,
        "gate_max_fcr": 0.10,
        "gate_min_coverage": 0.15,
        "gate_min_accepted": 8,
        "budget": 21,
        "m": 15,
    },
    {
        "mode_label": "topk16_m15_rule_t03_ba070_fcr010",
        "display_name": "Top-k 16, target 0.3, rule gate: calib BA >= 0.70 and FCR <= 0.10",
        "candidate": "topk_effect_k16_m15_target0.3",
        "gate_min_ba": 0.70,
        "gate_max_fcr": 0.10,
        "gate_min_coverage": 0.15,
        "gate_min_accepted": 8,
        "budget": 16,
        "m": 15,
    },
    {
        "mode_label": "topk16_m15_rule_ba075_fcr010",
        "display_name": "Top-k 16, rule gate: calib BA >= 0.75 and FCR <= 0.10",
        "candidate": "topk_effect_k16_m15_target0.4",
        "gate_min_ba": 0.75,
        "gate_max_fcr": 0.10,
        "gate_min_coverage": 0.15,
        "gate_min_accepted": 8,
        "budget": 16,
        "m": 15,
    },
)

META_GATE_SPECS = (
    {
        "mode_label": "topk21_m15_t03_meta_p060",
        "display_name": "Top-k 21, target 0.3, meta-readiness gate p >= 0.60",
        "candidate": "topk_effect_k21_m15_target0.3",
        "threshold": 0.60,
        "budget": 21,
        "m": 15,
    },
)

ADAPTIVE_RULE_SPECS = (
    {
        "mode_label": "adaptive_rule_ba060_fcr010_cov015",
        "display_name": "Adaptive rule selector: BA >= 0.60, FCR <= 0.10",
        "gate_min_ba": 0.60,
        "gate_max_fcr": 0.10,
        "gate_min_coverage": 0.15,
        "gate_min_accepted": 8,
    },
)

ADAPTIVE_META_SPECS = (
    {
        "mode_label": "adaptive_meta_p060",
        "display_name": "Adaptive meta selector p >= 0.60",
        "threshold": 0.60,
    },
)

COUNT_COLUMNS = [
    "all_folds",
    "deployed_folds",
    "all_total_trials",
    "all_total_rest",
    "all_total_intent",
    "deployed_total_trials",
    "deployed_total_rest",
    "deployed_total_intent",
    "n_accepted",
    "tp",
    "tn",
    "fp",
    "fn",
    "deployed_budget_sum",
    "deployed_calibration_sum",
]

BOOTSTRAP_METRICS = [
    "deployment_rate",
    "deployed_actual_coverage",
    "unconditional_actual_coverage",
    "accepted_accuracy",
    "accepted_balanced_accuracy",
    "accepted_sensitivity",
    "accepted_specificity",
    "false_command_rate_per_rest_opportunity",
    "unconditional_false_command_rate",
    "wrong_intent_rate_per_intent_opportunity",
    "intent_command_acceptance_rate",
    "unconditional_intent_command_acceptance_rate",
    "mean_deployed_channel_budget",
    "mean_deployed_calibration_trials",
]

PAIRED_COMPARISONS = (
    ("topk16_m15_no_gate", "all64_m15_no_gate"),
    ("topk21_m12_no_gate", "all64_m15_no_gate"),
    ("topk21_m15_no_gate", "all64_m15_no_gate"),
    ("central21_m15_no_gate", "all64_m15_no_gate"),
    ("fixed_motor21_m15_no_gate", "all64_m15_no_gate"),
    ("topk21_m15_rule_t03_ba065_fcr010", "all64_m15_no_gate"),
    ("topk21_m15_rule_t03_ba065_fcr010", "topk21_m15_no_gate"),
    ("topk16_m15_rule_t03_ba070_fcr010", "all64_m15_no_gate"),
    ("topk16_m15_rule_t03_ba070_fcr010", "topk16_m15_no_gate"),
    ("topk16_m15_rule_ba075_fcr010", "topk16_m15_no_gate"),
    ("topk21_m15_t03_meta_p060", "topk21_m15_no_gate"),
    ("adaptive_rule_ba060_fcr010_cov015", "topk16_m15_no_gate"),
    ("adaptive_meta_p060", "topk16_m15_no_gate"),
)

PAIRED_METRICS = [
    "accepted_balanced_accuracy",
    "unconditional_false_command_rate",
    "unconditional_actual_coverage",
    "deployment_rate",
]


def report_date() -> str:
    return datetime.now().date().isoformat()


def safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def subject_row_from_counts(
    *,
    mode_label: str,
    display_name: str,
    source: str,
    subject: str,
    all_folds: int,
    deployed_folds: int,
    all_total_trials: int,
    all_total_rest: int,
    all_total_intent: int,
    deployed_total_trials: int,
    deployed_total_rest: int,
    deployed_total_intent: int,
    n_accepted: int,
    tp: int,
    tn: int,
    fp: int,
    fn: int,
    deployed_budget_sum: float,
    deployed_calibration_sum: float,
) -> dict[str, float | int | str]:
    accepted_sensitivity = safe_rate(tp, tp + fn)
    accepted_specificity = safe_rate(tn, tn + fp)
    accepted_ba = (
        float("nan")
        if math.isnan(accepted_sensitivity) or math.isnan(accepted_specificity)
        else 0.5 * (accepted_sensitivity + accepted_specificity)
    )
    return {
        "mode_label": mode_label,
        "display_name": display_name,
        "source": source,
        "subject": subject,
        "all_folds": all_folds,
        "deployed_folds": deployed_folds,
        "deployment_rate": safe_rate(deployed_folds, all_folds),
        "all_total_trials": all_total_trials,
        "all_total_rest": all_total_rest,
        "all_total_intent": all_total_intent,
        "deployed_total_trials": deployed_total_trials,
        "deployed_total_rest": deployed_total_rest,
        "deployed_total_intent": deployed_total_intent,
        "n_accepted": n_accepted,
        "deployed_actual_coverage": safe_rate(n_accepted, deployed_total_trials),
        "unconditional_actual_coverage": safe_rate(n_accepted, all_total_trials),
        "accepted_accuracy": safe_rate(tp + tn, n_accepted),
        "accepted_balanced_accuracy": accepted_ba,
        "accepted_sensitivity": accepted_sensitivity,
        "accepted_specificity": accepted_specificity,
        "false_command_rate_per_rest_opportunity": safe_rate(fp, deployed_total_rest),
        "unconditional_false_command_rate": safe_rate(fp, all_total_rest),
        "wrong_intent_rate_per_intent_opportunity": safe_rate(fn, deployed_total_intent),
        "intent_command_acceptance_rate": safe_rate(tp, deployed_total_intent),
        "unconditional_intent_command_acceptance_rate": safe_rate(tp, all_total_intent),
        "mean_deployed_channel_budget": safe_rate(deployed_budget_sum, deployed_folds),
        "mean_deployed_calibration_trials": safe_rate(
            deployed_calibration_sum, deployed_folds
        ),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "deployed_budget_sum": deployed_budget_sum,
        "deployed_calibration_sum": deployed_calibration_sum,
    }


def aggregate_counts(rows: pd.DataFrame) -> dict[str, float | int]:
    sums = {col: float(rows[col].sum()) for col in COUNT_COLUMNS}
    tp = int(sums["tp"])
    tn = int(sums["tn"])
    fp = int(sums["fp"])
    fn = int(sums["fn"])
    n_accepted = int(sums["n_accepted"])
    accepted_sensitivity = safe_rate(tp, tp + fn)
    accepted_specificity = safe_rate(tn, tn + fp)
    accepted_ba = (
        float("nan")
        if math.isnan(accepted_sensitivity) or math.isnan(accepted_specificity)
        else 0.5 * (accepted_sensitivity + accepted_specificity)
    )
    return {
        "n_subjects": int(rows["subject"].nunique()),
        "all_folds": int(sums["all_folds"]),
        "deployed_folds": int(sums["deployed_folds"]),
        "deployment_rate": safe_rate(sums["deployed_folds"], sums["all_folds"]),
        "all_total_trials": int(sums["all_total_trials"]),
        "all_total_rest": int(sums["all_total_rest"]),
        "all_total_intent": int(sums["all_total_intent"]),
        "deployed_total_trials": int(sums["deployed_total_trials"]),
        "deployed_total_rest": int(sums["deployed_total_rest"]),
        "deployed_total_intent": int(sums["deployed_total_intent"]),
        "n_accepted": n_accepted,
        "deployed_actual_coverage": safe_rate(
            n_accepted, sums["deployed_total_trials"]
        ),
        "unconditional_actual_coverage": safe_rate(
            n_accepted, sums["all_total_trials"]
        ),
        "accepted_accuracy": safe_rate(tp + tn, n_accepted),
        "accepted_balanced_accuracy": accepted_ba,
        "accepted_sensitivity": accepted_sensitivity,
        "accepted_specificity": accepted_specificity,
        "false_command_rate_per_rest_opportunity": safe_rate(
            fp, sums["deployed_total_rest"]
        ),
        "unconditional_false_command_rate": safe_rate(fp, sums["all_total_rest"]),
        "wrong_intent_rate_per_intent_opportunity": safe_rate(
            fn, sums["deployed_total_intent"]
        ),
        "intent_command_acceptance_rate": safe_rate(tp, sums["deployed_total_intent"]),
        "unconditional_intent_command_acceptance_rate": safe_rate(
            tp, sums["all_total_intent"]
        ),
        "mean_deployed_channel_budget": safe_rate(
            sums["deployed_budget_sum"], sums["deployed_folds"]
        ),
        "mean_deployed_calibration_trials": safe_rate(
            sums["deployed_calibration_sum"], sums["deployed_folds"]
        ),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def mode_subject_template(
    mode_label: str, display_name: str, source: str, subjects: list[str]
) -> dict[str, dict[str, float | int | str]]:
    return {
        subject: subject_row_from_counts(
            mode_label=mode_label,
            display_name=display_name,
            source=source,
            subject=subject,
            all_folds=0,
            deployed_folds=0,
            all_total_trials=0,
            all_total_rest=0,
            all_total_intent=0,
            deployed_total_trials=0,
            deployed_total_rest=0,
            deployed_total_intent=0,
            n_accepted=0,
            tp=0,
            tn=0,
            fp=0,
            fn=0,
            deployed_budget_sum=0.0,
            deployed_calibration_sum=0.0,
        )
        for subject in subjects
    }


def build_no_gate_subject_rows(
    trial_df: pd.DataFrame, spec: dict[str, object]
) -> list[dict[str, float | int | str]]:
    mask = (
        (trial_df["policy"] == spec["policy"])
        & (trial_df["budget"] == spec["budget"])
        & (trial_df["m_per_class_per_run"] == spec["m"])
        & np.isclose(trial_df["target_coverage"], spec["target"])
    )
    df = trial_df[mask].copy()
    if df.empty:
        raise ValueError(f"No rows found for {spec['mode_label']}")

    rows: list[dict[str, float | int | str]] = []
    for subject, group in df.groupby("subject", sort=True):
        accepted = group["accepted"] == 1
        y_true = group["y_true"]
        y_pred = group["y_pred"]
        tp = int(((accepted) & (y_true == 1) & (y_pred == 1)).sum())
        tn = int(((accepted) & (y_true == 0) & (y_pred == 0)).sum())
        fp = int(((accepted) & (y_true == 0) & (y_pred == 1)).sum())
        fn = int(((accepted) & (y_true == 1) & (y_pred == 0)).sum())
        all_folds = int(group["test_run"].nunique())
        rows.append(
            subject_row_from_counts(
                mode_label=str(spec["mode_label"]),
                display_name=str(spec["display_name"]),
                source="no_gate_trial_results",
                subject=str(subject),
                all_folds=all_folds,
                deployed_folds=all_folds,
                all_total_trials=len(group),
                all_total_rest=int((y_true == 0).sum()),
                all_total_intent=int((y_true == 1).sum()),
                deployed_total_trials=len(group),
                deployed_total_rest=int((y_true == 0).sum()),
                deployed_total_intent=int((y_true == 1).sum()),
                n_accepted=int(accepted.sum()),
                tp=tp,
                tn=tn,
                fp=fp,
                fn=fn,
                deployed_budget_sum=float(spec["budget"]) * all_folds,
                deployed_calibration_sum=float(int(spec["m"]) * 4) * all_folds,
            )
        )
    return rows


def fold_totals_by_subject(fold_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    unique_folds = (
        fold_df.sort_values(["subject", "outer_test_run"])
        .groupby(["subject", "outer_test_run"], as_index=False)
        .first()
    )
    totals: dict[str, dict[str, int]] = {}
    for subject, group in unique_folds.groupby("subject", sort=True):
        totals[str(subject)] = {
            "all_folds": int(len(group)),
            "all_total_trials": int(group["test_total_trials"].sum()),
            "all_total_rest": int(group["test_total_rest"].sum()),
            "all_total_intent": int(group["test_total_intent"].sum()),
        }
    return totals


def counts_from_fold_group(group: pd.DataFrame) -> dict[str, int]:
    return {
        "deployed_folds": int(len(group)),
        "deployed_total_trials": int(group["test_total_trials"].sum()),
        "deployed_total_rest": int(group["test_total_rest"].sum()),
        "deployed_total_intent": int(group["test_total_intent"].sum()),
        "n_accepted": int(group["test_n_accepted"].sum()),
        "tp": int(group["test_tp"].sum()),
        "tn": int(group["test_tn"].sum()),
        "fp": int(group["test_fp"].sum()),
        "fn": int(group["test_fn"].sum()),
    }


def build_fixed_gate_subject_rows(
    fold_df: pd.DataFrame,
    spec: dict[str, object],
    *,
    source: str = "rule_readiness_gate",
) -> list[dict[str, float | int | str]]:
    candidate = fold_df[fold_df["candidate"] == spec["candidate"]].copy()
    if candidate.empty:
        raise ValueError(f"No rows found for {spec['mode_label']}")
    totals = fold_totals_by_subject(candidate)
    subjects = sorted(totals)
    rows = mode_subject_template(
        str(spec["mode_label"]), str(spec["display_name"]), source, subjects
    )
    deploy_mask = (
        (candidate["calib_accepted_balanced_accuracy"] >= spec["gate_min_ba"])
        & (
            candidate["calib_false_command_rate_per_rest_opportunity"]
            <= spec["gate_max_fcr"]
        )
        & (candidate["calib_actual_coverage"] >= spec["gate_min_coverage"])
        & (candidate["calib_n_accepted"] >= spec["gate_min_accepted"])
    )
    deployed = candidate[deploy_mask]
    for subject in subjects:
        subject_deployed = deployed[deployed["subject"] == subject]
        counts = counts_from_fold_group(subject_deployed)
        total = totals[subject]
        rows[subject] = subject_row_from_counts(
            mode_label=str(spec["mode_label"]),
            display_name=str(spec["display_name"]),
            source=source,
            subject=subject,
            all_folds=total["all_folds"],
            deployed_folds=counts["deployed_folds"],
            all_total_trials=total["all_total_trials"],
            all_total_rest=total["all_total_rest"],
            all_total_intent=total["all_total_intent"],
            deployed_total_trials=counts["deployed_total_trials"],
            deployed_total_rest=counts["deployed_total_rest"],
            deployed_total_intent=counts["deployed_total_intent"],
            n_accepted=counts["n_accepted"],
            tp=counts["tp"],
            tn=counts["tn"],
            fp=counts["fp"],
            fn=counts["fn"],
            deployed_budget_sum=float(spec["budget"]) * counts["deployed_folds"],
            deployed_calibration_sum=float(int(spec["m"]) * 4)
            * counts["deployed_folds"],
        )
    return list(rows.values())


def build_meta_gate_subject_rows(
    meta_df: pd.DataFrame, spec: dict[str, object]
) -> list[dict[str, float | int | str]]:
    candidate = meta_df[meta_df["candidate"] == spec["candidate"]].copy()
    if candidate.empty:
        raise ValueError(f"No rows found for {spec['mode_label']}")
    totals = fold_totals_by_subject(candidate)
    subjects = sorted(totals)
    rows = mode_subject_template(
        str(spec["mode_label"]), str(spec["display_name"]), "meta_readiness_gate", subjects
    )
    deployed = candidate[candidate["meta_success_probability"] >= spec["threshold"]]
    for subject in subjects:
        subject_deployed = deployed[deployed["subject"] == subject]
        counts = counts_from_fold_group(subject_deployed)
        total = totals[subject]
        rows[subject] = subject_row_from_counts(
            mode_label=str(spec["mode_label"]),
            display_name=str(spec["display_name"]),
            source="meta_readiness_gate",
            subject=subject,
            all_folds=total["all_folds"],
            deployed_folds=counts["deployed_folds"],
            all_total_trials=total["all_total_trials"],
            all_total_rest=total["all_total_rest"],
            all_total_intent=total["all_total_intent"],
            deployed_total_trials=counts["deployed_total_trials"],
            deployed_total_rest=counts["deployed_total_rest"],
            deployed_total_intent=counts["deployed_total_intent"],
            n_accepted=counts["n_accepted"],
            tp=counts["tp"],
            tn=counts["tn"],
            fp=counts["fp"],
            fn=counts["fn"],
            deployed_budget_sum=float(spec["budget"]) * counts["deployed_folds"],
            deployed_calibration_sum=float(int(spec["m"]) * 4)
            * counts["deployed_folds"],
        )
    return list(rows.values())


def build_adaptive_subject_rows(
    fold_df: pd.DataFrame,
    assignment_df: pd.DataFrame,
    spec: dict[str, object],
    *,
    source: str,
) -> list[dict[str, float | int | str]]:
    totals = fold_totals_by_subject(fold_df)
    subjects = sorted(totals)
    rows = mode_subject_template(
        str(spec["mode_label"]), str(spec["display_name"]), source, subjects
    )
    if source == "adaptive_rule_selector":
        assignment_mask = (
            np.isclose(assignment_df["gate_min_calib_ba"], spec["gate_min_ba"])
            & np.isclose(assignment_df["gate_max_calib_fcr"], spec["gate_max_fcr"])
            & np.isclose(
                assignment_df["gate_min_calib_coverage"], spec["gate_min_coverage"]
            )
        )
    elif source == "adaptive_meta_selector":
        assignment_mask = np.isclose(assignment_df["threshold"], spec["threshold"])
    else:
        raise ValueError(source)
    selected = assignment_df[assignment_mask].copy()
    if selected.empty:
        raise ValueError(f"No adaptive assignments found for {spec['mode_label']}")
    selected = selected.merge(
        fold_df,
        on=["subject", "outer_test_run", "candidate"],
        how="left",
        suffixes=("_assigned", ""),
    )
    if selected["test_total_trials"].isna().any():
        raise ValueError(f"Adaptive assignment join failed for {spec['mode_label']}")
    for subject in subjects:
        subject_selected = selected[selected["subject"] == subject]
        counts = counts_from_fold_group(subject_selected)
        total = totals[subject]
        rows[subject] = subject_row_from_counts(
            mode_label=str(spec["mode_label"]),
            display_name=str(spec["display_name"]),
            source=source,
            subject=subject,
            all_folds=total["all_folds"],
            deployed_folds=counts["deployed_folds"],
            all_total_trials=total["all_total_trials"],
            all_total_rest=total["all_total_rest"],
            all_total_intent=total["all_total_intent"],
            deployed_total_trials=counts["deployed_total_trials"],
            deployed_total_rest=counts["deployed_total_rest"],
            deployed_total_intent=counts["deployed_total_intent"],
            n_accepted=counts["n_accepted"],
            tp=counts["tp"],
            tn=counts["tn"],
            fp=counts["fp"],
            fn=counts["fn"],
            deployed_budget_sum=float(subject_selected["budget"].sum()),
            deployed_calibration_sum=float(
                subject_selected["nominal_calibration_trials"].sum()
            ),
        )
    return list(rows.values())


def bootstrap_ci(
    subject_metrics: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    for mode_label, group in subject_metrics.groupby("mode_label", sort=False):
        group = group.reset_index(drop=True)
        point = aggregate_counts(group)
        values = {metric: [] for metric in BOOTSTRAP_METRICS}
        n_subjects = len(group)
        for _ in range(n_bootstrap):
            sample_idx = rng.integers(0, n_subjects, size=n_subjects)
            stats = aggregate_counts(group.iloc[sample_idx])
            for metric in BOOTSTRAP_METRICS:
                values[metric].append(float(stats[metric]))
        for metric in BOOTSTRAP_METRICS:
            samples = np.array(values[metric], dtype=float)
            finite = samples[np.isfinite(samples)]
            low, high = (
                (float("nan"), float("nan"))
                if len(finite) == 0
                else np.quantile(finite, [0.025, 0.975])
            )
            rows.append(
                {
                    "mode_label": mode_label,
                    "display_name": str(group["display_name"].iloc[0]),
                    "source": str(group["source"].iloc[0]),
                    "metric": metric,
                    "point_estimate": point[metric],
                    "ci_low": low,
                    "ci_high": high,
                    "n_subjects": point["n_subjects"],
                    "n_bootstrap": n_bootstrap,
                }
            )
    return pd.DataFrame(rows)


def holm_adjust(p_values: pd.Series) -> pd.Series:
    adjusted = pd.Series(index=p_values.index, dtype=float)
    finite = p_values.dropna().sort_values()
    m = len(finite)
    running_max = 0.0
    for rank, (idx, p_value) in enumerate(finite.items(), start=1):
        adj = min(1.0, (m - rank + 1) * float(p_value))
        running_max = max(running_max, adj)
        adjusted.loc[idx] = running_max
    return adjusted


def paired_bootstrap_ci(
    diffs: np.ndarray, *, rng: np.random.Generator, n_bootstrap: int
) -> tuple[float, float]:
    finite = diffs[np.isfinite(diffs)]
    if len(finite) == 0:
        return float("nan"), float("nan")
    samples = []
    for _ in range(n_bootstrap):
        sample = rng.choice(finite, size=len(finite), replace=True)
        samples.append(float(np.mean(sample)))
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def paired_comparisons(
    subject_metrics: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    by_mode = {
        mode: group.set_index("subject")
        for mode, group in subject_metrics.groupby("mode_label", sort=False)
    }
    for challenger, comparator in PAIRED_COMPARISONS:
        missing = [mode for mode in (challenger, comparator) if mode not in by_mode]
        if missing:
            raise ValueError(
                "Missing mode(s) for paired comparison: " + ", ".join(missing)
            )
        joined = by_mode[challenger].join(
            by_mode[comparator],
            lsuffix="_challenger",
            rsuffix="_comparator",
            how="inner",
        )
        for metric in PAIRED_METRICS:
            challenger_values = joined[f"{metric}_challenger"].astype(float).to_numpy()
            comparator_values = joined[f"{metric}_comparator"].astype(float).to_numpy()
            diffs = challenger_values - comparator_values
            finite = diffs[np.isfinite(diffs)]
            if len(finite) == 0:
                continue
            nonzero = finite[np.abs(finite) > 1e-12]
            if len(nonzero) == 0:
                wilcoxon_p = 1.0
                sign_p = 1.0
            else:
                wilcoxon_p = float(wilcoxon(nonzero, alternative="two-sided").pvalue)
                positives = int((nonzero > 0).sum())
                sign_p = float(binomtest(positives, len(nonzero)).pvalue)
            ci_low, ci_high = paired_bootstrap_ci(
                finite, rng=rng, n_bootstrap=n_bootstrap
            )
            rows.append(
                {
                    "challenger": challenger,
                    "comparator": comparator,
                    "metric": metric,
                    "n_pairs": int(len(finite)),
                    "mean_challenger": float(np.nanmean(challenger_values)),
                    "mean_comparator": float(np.nanmean(comparator_values)),
                    "mean_difference": float(np.mean(finite)),
                    "median_difference": float(np.median(finite)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_positive": int((finite > 0).sum()),
                    "n_negative": int((finite < 0).sum()),
                    "wilcoxon_p": wilcoxon_p,
                    "sign_test_p": sign_p,
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["wilcoxon_p_holm_by_metric"] = result.groupby("metric")[
            "wilcoxon_p"
        ].transform(holm_adjust)
    return result


def subject_tail_summary(subject_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for mode_label, group in subject_metrics.groupby("mode_label", sort=False):
        for metric in [
            "accepted_balanced_accuracy",
            "unconditional_false_command_rate",
            "unconditional_actual_coverage",
            "deployment_rate",
        ]:
            values = group[metric].astype(float).to_numpy()
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "mode_label": mode_label,
                    "display_name": str(group["display_name"].iloc[0]),
                    "metric": metric,
                    "n_finite_subjects": int(len(finite)),
                    "mean": float(np.mean(finite)) if len(finite) else float("nan"),
                    "q10": float(np.quantile(finite, 0.10))
                    if len(finite)
                    else float("nan"),
                    "median": float(np.median(finite)) if len(finite) else float("nan"),
                    "q90": float(np.quantile(finite, 0.90))
                    if len(finite)
                    else float("nan"),
                    "min": float(np.min(finite)) if len(finite) else float("nan"),
                    "max": float(np.max(finite)) if len(finite) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def write_report(
    out_path: Path,
    aggregate_ci: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> None:
    key_metrics = [
        "deployment_rate",
        "deployed_actual_coverage",
        "accepted_balanced_accuracy",
        "false_command_rate_per_rest_opportunity",
        "unconditional_actual_coverage",
        "unconditional_false_command_rate",
    ]
    pivot = aggregate_ci[aggregate_ci["metric"].isin(key_metrics)].copy()

    def fmt_ci(row: pd.Series) -> str:
        return (
            f"{row['point_estimate']:.3f} "
            f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
        )

    metric_table = pivot.pivot_table(
        index=["mode_label", "display_name"],
        columns="metric",
        values=["point_estimate", "ci_low", "ci_high"],
        aggfunc="first",
    )
    rows: list[str] = []
    rows.append("# Milestone 06: Statistical Validation\n")
    rows.append(f"Date: {report_date()}\n")
    rows.append("## Validation Unit\n")
    rows.append(
        "All confidence intervals and paired tests use subject-level aggregation "
        "over the 105-subject clean cohort. Bootstrap intervals resample subjects "
        "with replacement. Readiness policies are evaluated both conditionally on "
        "deployed sessions and unconditionally over all held-out opportunities.\n"
    )
    rows.append("## Bootstrap Confidence Intervals\n")
    rows.append(
        "| Policy | Deploy | Deployed Cov. | Accepted BA | Deployed FCR | Uncond. Cov. | Uncond. FCR |"
    )
    rows.append("|---|---:|---:|---:|---:|---:|---:|")
    ordered_modes = aggregate_ci["mode_label"].drop_duplicates().tolist()
    for mode in ordered_modes:
        mode_rows = pivot[pivot["mode_label"] == mode].set_index("metric")
        display = str(mode_rows["display_name"].iloc[0])
        cells = []
        for metric in key_metrics:
            cells.append(fmt_ci(mode_rows.loc[metric]))
        rows.append(f"| {display} | " + " | ".join(cells) + " |")

    rows.append("\n## Key Paired Comparisons\n")
    rows.append(
        "| Challenger vs Comparator | Metric | Mean Diff. | 95% CI | Wilcoxon p (Holm) | n |"
    )
    rows.append("|---|---|---:|---:|---:|---:|")
    if comparisons.empty:
        rows.append("| No comparisons generated |  |  |  |  |  |")
    else:
        priority_keys = {
            (
                "topk16_m15_no_gate",
                "all64_m15_no_gate",
                "accepted_balanced_accuracy",
            ),
            (
                "topk16_m15_no_gate",
                "all64_m15_no_gate",
                "unconditional_false_command_rate",
            ),
            (
                "topk16_m15_no_gate",
                "all64_m15_no_gate",
                "unconditional_actual_coverage",
            ),
            (
                "topk21_m15_no_gate",
                "all64_m15_no_gate",
                "unconditional_false_command_rate",
            ),
            (
                "topk21_m15_rule_t03_ba065_fcr010",
                "all64_m15_no_gate",
                "unconditional_false_command_rate",
            ),
            (
                "topk21_m15_rule_t03_ba065_fcr010",
                "all64_m15_no_gate",
                "unconditional_actual_coverage",
            ),
            (
                "topk21_m15_rule_t03_ba065_fcr010",
                "all64_m15_no_gate",
                "deployment_rate",
            ),
            (
                "topk21_m15_rule_t03_ba065_fcr010",
                "topk21_m15_no_gate",
                "unconditional_false_command_rate",
            ),
            (
                "topk21_m15_rule_t03_ba065_fcr010",
                "topk21_m15_no_gate",
                "unconditional_actual_coverage",
            ),
            (
                "topk16_m15_rule_t03_ba070_fcr010",
                "all64_m15_no_gate",
                "deployment_rate",
            ),
            (
                "topk21_m15_t03_meta_p060",
                "topk21_m15_no_gate",
                "unconditional_false_command_rate",
            ),
            (
                "topk21_m15_t03_meta_p060",
                "topk21_m15_no_gate",
                "unconditional_actual_coverage",
            ),
            (
                "topk21_m15_t03_meta_p060",
                "topk21_m15_no_gate",
                "deployment_rate",
            ),
        }
        priority = comparisons[
            comparisons.apply(
                lambda row: (
                    row["challenger"],
                    row["comparator"],
                    row["metric"],
                )
                in priority_keys,
                axis=1,
            )
        ].copy()
        for _, row in priority.iterrows():
            rows.append(
                "| "
                f"{row['challenger']} vs {row['comparator']} | "
                f"{row['metric']} | "
                f"{row['mean_difference']:.3f} | "
                f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}] | "
                f"{row['wilcoxon_p_holm_by_metric']:.3g} | "
                f"{int(row['n_pairs'])} |"
            )

    rows.append("\n## Non-Inferiority Check\n")
    rows.append(
        "For no-gate channel-budgeted policies, use a 0.05 balanced-accuracy "
        "non-inferiority margin against the 64-channel baseline. This is an "
        "engineering margin for screening, not yet a final clinical margin.\n"
    )
    rows.append("| Policy vs All-Channel | Mean BA Diff. | 95% CI | Non-Inferior at 0.05? |")
    rows.append("|---|---:|---:|---:|")
    if comparisons.empty:
        rows.append("| No comparisons generated |  |  |  |")
    else:
        ni = comparisons[
            (comparisons["comparator"] == "all64_m15_no_gate")
            & (comparisons["metric"] == "accepted_balanced_accuracy")
        ]
        for _, row in ni.iterrows():
            is_noninferior = row["ci_low"] > -0.05
            rows.append(
                "| "
                f"{row['challenger']} | "
                f"{row['mean_difference']:.3f} | "
                f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}] | "
                f"{'yes' if is_noninferior else 'no'} |"
            )

    rows.append("\n## Interpretation\n")
    rows.append(
        "The defensible claim is a decision-analytic one: a post-cue 0.5-3.5 s "
        "window with budgeted top-k personalization and calibration readiness "
        "gating can improve the practical reliability-safety-access frontier. "
        "The primary TopK21 rule reduces unconditional false-command exposure "
        "relative to always-deploy baselines while retaining moderate deployment. "
        "It should still be framed as cue-paced command gating, not clinical "
        "effectiveness or pure motor-cortex decoding.\n"
    )
    out_path.write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Subject-level statistical validation for risk-aware BMI policies."
    )
    parser.add_argument("--config", default="configs/local.json")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260607)
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    frontier_dir = (
        cfg.run_root
        / "calibration_frontier"
        / "calibration_frontier_logvar_intent_rest_t0.5-3.5"
    )
    readiness_dir = (
        cfg.run_root
        / "readiness_gate"
        / "readiness_gate_logvar_intent_rest_t0.5-3.5_target_sweep"
    )
    meta_dir = readiness_dir / "meta_readiness"

    trial_df = pd.read_csv(frontier_dir / "trial_results.csv")
    fold_df = pd.read_csv(readiness_dir / "fold_results.csv")
    rule_assignments = pd.read_csv(readiness_dir / "selected_policy_assignments.csv")
    meta_df = pd.read_csv(meta_dir / "meta_predictions.csv")
    meta_assignments = pd.read_csv(meta_dir / "meta_selector_assignments.csv")

    subject_rows: list[dict[str, float | int | str]] = []
    for spec in NO_GATE_SPECS:
        subject_rows.extend(build_no_gate_subject_rows(trial_df, spec))
    for spec in RULE_GATE_SPECS:
        subject_rows.extend(build_fixed_gate_subject_rows(fold_df, spec))
    for spec in META_GATE_SPECS:
        subject_rows.extend(build_meta_gate_subject_rows(meta_df, spec))
    for spec in ADAPTIVE_RULE_SPECS:
        subject_rows.extend(
            build_adaptive_subject_rows(
                fold_df, rule_assignments, spec, source="adaptive_rule_selector"
            )
        )
    for spec in ADAPTIVE_META_SPECS:
        subject_rows.extend(
            build_adaptive_subject_rows(
                meta_df, meta_assignments, spec, source="adaptive_meta_selector"
            )
        )

    subject_metrics = pd.DataFrame(subject_rows)
    aggregate_rows = []
    for mode_label, group in subject_metrics.groupby("mode_label", sort=False):
        row = {
            "mode_label": mode_label,
            "display_name": str(group["display_name"].iloc[0]),
            "source": str(group["source"].iloc[0]),
        }
        row.update(aggregate_counts(group))
        aggregate_rows.append(row)
    aggregate_summary = pd.DataFrame(aggregate_rows)
    aggregate_ci = bootstrap_ci(
        subject_metrics, n_bootstrap=args.n_bootstrap, seed=args.seed
    )
    comparisons = paired_comparisons(
        subject_metrics, n_bootstrap=args.n_bootstrap, seed=args.seed + 1
    )
    tails = subject_tail_summary(subject_metrics)

    out_dir = cfg.run_root / "validation" / "subject_level_policy_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    subject_metrics.to_csv(out_dir / "subject_metrics.csv", index=False)
    aggregate_summary.to_csv(out_dir / "aggregate_summary.csv", index=False)
    aggregate_ci.to_csv(out_dir / "bootstrap_confidence_intervals.csv", index=False)
    comparisons.to_csv(out_dir / "paired_comparisons.csv", index=False)
    tails.to_csv(out_dir / "subject_tail_summary.csv", index=False)
    write_report(out_dir / "milestone_06_statistical_validation.md", aggregate_ci, comparisons)

    display = aggregate_ci[
        aggregate_ci["metric"].isin(
            [
                "deployment_rate",
                "deployed_actual_coverage",
                "accepted_balanced_accuracy",
                "false_command_rate_per_rest_opportunity",
                "unconditional_actual_coverage",
                "unconditional_false_command_rate",
            ]
        )
    ].copy()
    print(f"output_dir={out_dir}")
    print(f"subject_metrics={out_dir / 'subject_metrics.csv'}")
    print(f"aggregate_summary={out_dir / 'aggregate_summary.csv'}")
    print(f"bootstrap_confidence_intervals={out_dir / 'bootstrap_confidence_intervals.csv'}")
    print(f"paired_comparisons={out_dir / 'paired_comparisons.csv'}")
    print(
        display.pivot_table(
            index="mode_label",
            columns="metric",
            values="point_estimate",
            aggfunc="first",
        ).to_string(float_format=lambda x: f"{x:.3f}")
    )


if __name__ == "__main__":
    main()
