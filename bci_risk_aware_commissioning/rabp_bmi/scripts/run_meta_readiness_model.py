from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "budget",
    "m_per_class_per_run",
    "nominal_calibration_trials",
    "actual_calibration_trials",
    "target_coverage",
    "calibrated_threshold",
    "n_selected",
    "calib_total_trials",
    "calib_total_rest",
    "calib_total_intent",
    "calib_n_accepted",
    "calib_accepted_rest",
    "calib_accepted_intent",
    "calib_actual_coverage",
    "calib_accepted_accuracy",
    "calib_accepted_balanced_accuracy",
    "calib_accepted_sensitivity",
    "calib_accepted_specificity",
    "calib_false_command_rate_per_rest_opportunity",
    "calib_wrong_rest_rate_per_intent_opportunity",
    "calib_intent_command_acceptance_rate",
    "calib_tp",
    "calib_tn",
    "calib_fp",
    "calib_fn",
]

CATEGORICAL_FEATURES = ["policy"]


def make_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Subject-held-out meta-model for calibration readiness."
    )
    parser.add_argument("fold_results_csv")
    parser.add_argument("--min-test-ba", type=float, default=0.80)
    parser.add_argument("--max-test-fcr", type=float, default=0.06)
    parser.add_argument("--min-test-coverage", type=float, default=0.20)
    args = parser.parse_args()

    path = Path(args.fold_results_csv)
    df = pd.read_csv(path)
    df["success_label"] = (
        (df["test_accepted_balanced_accuracy"] >= args.min_test_ba)
        & (df["test_false_command_rate_per_rest_opportunity"] <= args.max_test_fcr)
        & (df["test_actual_coverage"] >= args.min_test_coverage)
    ).astype(int)

    predictions: list[pd.DataFrame] = []
    skipped_subjects: list[str] = []
    for subject in sorted(df["subject"].unique()):
        train = df[df["subject"] != subject]
        test = df[df["subject"] == subject]
        if train["success_label"].nunique() < 2:
            skipped_subjects.append(str(subject))
            continue
        clf = make_pipeline()
        clf.fit(train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train["success_label"])
        proba = clf.predict_proba(test[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
        pred = test.copy()
        pred["meta_success_probability"] = proba
        pred["meta_cv_subject"] = subject
        predictions.append(pred)

    if not predictions:
        raise RuntimeError("No predictions generated")

    pred_df = pd.concat(predictions, ignore_index=True)
    out_dir = path.parent / "meta_readiness"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "meta_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    y = pred_df["success_label"].to_numpy()
    p = pred_df["meta_success_probability"].to_numpy()
    roc_auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")
    ap = average_precision_score(y, p)

    metric_rows = [
        {
            "n_rows": len(pred_df),
            "n_subjects": pred_df["subject"].nunique(),
            "success_rate": float(pred_df["success_label"].mean()),
            "roc_auc": float(roc_auc),
            "average_precision": float(ap),
            "min_test_ba": args.min_test_ba,
            "max_test_fcr": args.max_test_fcr,
            "min_test_coverage": args.min_test_coverage,
            "skipped_subjects": " ".join(skipped_subjects),
        }
    ]
    metrics_path = out_dir / "meta_model_metrics.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)

    by_candidate = (
        pred_df.groupby("candidate")
        .agg(
            rows=("success_label", "size"),
            success_rate=("success_label", "mean"),
            mean_probability=("meta_success_probability", "mean"),
        )
        .reset_index()
    )
    by_candidate_path = out_dir / "meta_by_candidate.csv"
    by_candidate.to_csv(by_candidate_path, index=False)

    print(f"predictions={pred_path}")
    print(f"metrics={metrics_path}")
    print(f"by_candidate={by_candidate_path}")
    print(f"rows={len(pred_df)} subjects={pred_df['subject'].nunique()}")
    print(f"success_rate={pred_df['success_label'].mean():.3f}")
    print(f"roc_auc={roc_auc:.3f}")
    print(f"average_precision={ap:.3f}")
    print(by_candidate.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()

