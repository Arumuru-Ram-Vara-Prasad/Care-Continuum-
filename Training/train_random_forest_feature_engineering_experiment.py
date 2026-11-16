"""Controlled feature-engineering experiment for Care Continuum.

Compares the existing leakage-aware Random Forest feature set (baseline)
against the same pipeline plus two deterministic features:
  - medication_changes_count
  - has_specialty

Safeguards:
  - Reuses and validates the existing patient-level split assignments.
  - Uses GroupKFold by patient_nbr on training rows only.
  - Fits imputation, scaling, and one-hot encoding inside every CV fold.
  - Fits each final model on training rows and reports validation performance.
  - Does not tune against or score the held-out test set.
  - Saves outputs in a NEW folder so existing model results are not overwritten.

Place in <project>/Training/ and run from project root:
    python .\\Training\\train_random_forest_feature_engineering_experiment.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "readmitted_30d"
GROUP_COL = "patient_nbr"
NOT_MEASURED_FEATURES = ["max_glu_serum", "A1Cresult"]
NUMERIC_FEATURES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]
CATEGORICAL_CODE_FEATURES = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]
DRUG_FEATURES_TO_EXCLUDE = [
    "chlorpropamide",
    "acetohexamide",
    "tolbutamide",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "examide",
    "citoglipton",
]
MEDICATION_FEATURES_FOR_COUNT = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "glimepiride",
    "glipizide",
    "glyburide",
    "pioglitazone",
    "rosiglitazone",
    "insulin",
    "glyburide-metformin",
]
ENGINEERED_FEATURES = ["medication_changes_count", "has_specialty"]


def add_care_continuum_engineered_features(features: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with medication_changes_count and has_specialty added.

    medication_changes_count:
        Count values equal to 'Up' or 'Down' across the ten specified
        retained medication columns. Matching ignores leading/trailing spaces
        and letter case. Original medication columns are left unchanged.

    has_specialty:
        1 when medical_specialty_group contains a nonblank, non-'?' value
        other than 'Unknown' (case-insensitive); otherwise 0. Missing/blank
        specialty values count as 0 so missing data is not mistaken for a
        known specialist. The original specialty column is left unchanged.

    All existing columns, including 'change' and 'diabetesMed', are preserved.
    """
    required = [*MEDICATION_FEATURES_FOR_COUNT, "medical_specialty_group"]
    missing = [col for col in required if col not in features.columns]
    if missing:
        raise ValueError(
            "Cannot engineer the requested features; required columns are missing: "
            + ", ".join(missing)
        )

    out = features.copy()

    medication_values = out[MEDICATION_FEATURES_FOR_COUNT].apply(
        lambda col: col.astype("string").str.strip().str.casefold().isin(["up", "down"])
    )
    out["medication_changes_count"] = medication_values.sum(axis=1).astype("int8")

    specialty = out["medical_specialty_group"].astype("string").str.strip()
    known_specialty = (
        specialty.notna()
        & specialty.ne("")
        & specialty.ne("?")
        & specialty.str.casefold().ne("unknown")
    )
    out["has_specialty"] = known_specialty.fillna(False).astype("int8")
    return out


def find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "dataset").is_dir():
            return candidate
    raise FileNotFoundError("Cannot find project root: no parent folder contains 'dataset'.")


def locate_file(candidates, description):
    for path in candidates:
        if path.is_file():
            return path
    searched = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(f"Could not find {description}. Searched:\n{searched}")


def clean_text_series(series: pd.Series) -> pd.Series:
    out = series.map(lambda x: np.nan if pd.isna(x) else str(x).strip())
    return out.replace({"": np.nan, "?": np.nan})


def load_raw_features(raw_path: Path):
    df = pd.read_csv(
        raw_path,
        dtype={GROUP_COL: "string"},
        keep_default_na=False,
        low_memory=False,
    )
    required = [
        TARGET_COL,
        GROUP_COL,
        *NUMERIC_FEATURES,
        *CATEGORICAL_CODE_FEATURES,
        *MEDICATION_FEATURES_FOR_COUNT,
        "medical_specialty_group",
    ]
    missing_required = [col for col in required if col not in df.columns]
    if missing_required:
        raise ValueError(f"Required columns missing from pruned CSV: {missing_required}")

    # Preserve the existing preprocessing semantics for unmeasured lab results.
    for col in NOT_MEASURED_FEATURES:
        if col in df.columns:
            text = df[col].astype("string").str.strip()
            mask = text.isna() | text.eq("") | text.str.casefold().eq("none")
            df.loc[mask.fillna(False), col] = "Not measured"

    for col in df.columns:
        if col not in NOT_MEASURED_FEATURES and (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
        ):
            df[col] = clean_text_series(df[col])

    y_series = pd.to_numeric(df[TARGET_COL], errors="coerce")
    if y_series.isna().any() or not set(y_series.unique()).issubset({0, 1}):
        raise ValueError(f"{TARGET_COL} must contain only complete binary 0/1 labels.")
    y = y_series.astype("int8").to_numpy()

    groups_series = df[GROUP_COL].astype("string").str.strip()
    if groups_series.isna().any() or groups_series.eq("").any():
        raise ValueError("Missing/blank patient_nbr; cannot do patient-grouped CV.")
    groups = groups_series.astype(str).to_numpy()

    # Keep the existing exclusions and feature set. IDs and the target never
    # enter the model. The ten requested medications remain as original inputs.
    excluded = [TARGET_COL, GROUP_COL, "encounter_id", *DRUG_FEATURES_TO_EXCLUDE]
    X_baseline = df.drop(columns=[c for c in excluded if c in df.columns]).copy()
    for col in NUMERIC_FEATURES:
        X_baseline[col] = pd.to_numeric(X_baseline[col], errors="coerce")
    for col in CATEGORICAL_CODE_FEATURES:
        X_baseline[col] = clean_text_series(X_baseline[col])
    base_numeric_cols = list(NUMERIC_FEATURES)
    missing_numeric = [c for c in base_numeric_cols if c not in X_baseline.columns]
    if missing_numeric:
        raise ValueError(f"Numeric predictors missing after exclusions: {missing_numeric}")
    base_categorical_cols = [c for c in X_baseline.columns if c not in base_numeric_cols]
    for col in base_categorical_cols:
        X_baseline[col] = clean_text_series(X_baseline[col])

    # Feature engineering is deterministic and does not learn from target labels.
    # It runs on the predictors only; the original columns remain unchanged.
    X_engineered = add_care_continuum_engineered_features(X_baseline)
    eng_numeric_cols = [*base_numeric_cols, *ENGINEERED_FEATURES]
    eng_categorical_cols = [c for c in X_engineered.columns if c not in eng_numeric_cols]

    return df, X_baseline, X_engineered, y, groups, base_numeric_cols, base_categorical_cols, eng_numeric_cols, eng_categorical_cols


def make_pipeline(numeric_cols, categorical_cols):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ], remainder="drop", sparse_threshold=1.0, verbose_feature_names_out=True)

    classifier = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", classifier)])


def evaluate(y_true, probability, threshold, model_name, split_name):
    prediction = (probability >= threshold).astype(np.int8)
    cm = confusion_matrix(y_true, prediction, labels=[0, 1])
    metrics = {
        "model_variant": model_name,
        "evaluation_split": split_name,
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "average_precision_pr_auc": float(average_precision_score(y_true, probability)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "confusion_matrix_labels_0_then_1": cm.tolist(),
    }
    return metrics, prediction, cm


def best_f1_threshold(y_true, probability):
    """Exploratory validation-only threshold summary; not an unbiased estimate."""
    thresholds = np.linspace(0.01, 0.99, 197)
    rows = []
    for threshold in thresholds:
        pred = (probability >= threshold).astype(np.int8)
        rows.append({
            "threshold": float(threshold),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
        })
    frame = pd.DataFrame(rows)
    # If F1 ties, choose the higher threshold to reduce false-positive burden.
    max_f1 = frame["f1"].max()
    tied = frame[frame["f1"].eq(max_f1)]
    return tied.sort_values("threshold", ascending=False).iloc[0].to_dict(), frame


def main():
    root = find_project_root()
    dataset_dir = root / "dataset"
    raw_path = locate_file(
        sorted(dataset_dir.glob("diabetic_data_drugs_pruned*.csv"),
               key=lambda p: p.stat().st_mtime, reverse=True),
        "already-pruned input CSV (diabetic_data_drugs_pruned*.csv)",
    )
    assignments_path = locate_file([
        dataset_dir / "model_ready" / "split_assignments.csv",
        dataset_dir / "split_assignments.csv",
        root / "Preprocessing" / "split_assignments.csv",
        root / "split_assignments.csv",
    ], "split_assignments.csv")

    # New output directory: the existing baseline artifacts are not overwritten.
    out_dir = root / "models" / "random_forest_feature_engineering_experiment"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Raw cleaned/pruned input: {raw_path}")
    print(f"Existing split assignments: {assignments_path}")
    (raw_df, X_baseline, X_engineered, y, groups,
     base_numeric_cols, base_categorical_cols,
     eng_numeric_cols, eng_categorical_cols) = load_raw_features(raw_path)

    assignments = pd.read_csv(assignments_path, dtype={GROUP_COL: "string"})
    required_assignment_cols = {"source_row_index", GROUP_COL, TARGET_COL, "split"}
    if not required_assignment_cols.issubset(assignments.columns):
        raise ValueError(
            "split_assignments.csv missing required columns: "
            f"{sorted(required_assignment_cols - set(assignments.columns))}"
        )
    if assignments["source_row_index"].duplicated().any():
        raise ValueError("source_row_index must be unique in split_assignments.csv.")
    if len(assignments) != len(raw_df):
        raise ValueError(
            f"Row-count mismatch: pruned CSV has {len(raw_df)} rows, split assignments have "
            f"{len(assignments)}. Cannot safely reuse assignments; no model trained."
        )
    assignments = assignments.sort_values("source_row_index").reset_index(drop=True)
    expected_idx = np.arange(len(raw_df))
    actual_idx = pd.to_numeric(assignments["source_row_index"], errors="raise").to_numpy()
    if not np.array_equal(actual_idx, expected_idx):
        raise ValueError("source_row_index is not exactly 0..N-1; refusing to assume row alignment.")
    if not np.array_equal(pd.to_numeric(assignments[TARGET_COL], errors="raise").to_numpy().astype(int), y):
        raise ValueError("Target labels in split_assignments.csv do not match the pruned CSV.")
    if not np.array_equal(assignments[GROUP_COL].astype(str).to_numpy(), groups):
        raise ValueError("patient_nbr in split_assignments.csv does not match the pruned CSV.")
    split = assignments["split"].astype(str).str.lower().to_numpy()
    if not set(np.unique(split)).issubset({"train", "validation", "test"}):
        raise ValueError(f"Unexpected split labels: {sorted(set(split))}")

    train_idx = np.flatnonzero(split == "train")
    val_idx = np.flatnonzero(split == "validation")
    test_idx = np.flatnonzero(split == "test")
    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise ValueError("At least one train/validation/test split is empty.")
    group_sets = [set(groups[idx]) for idx in (train_idx, val_idx, test_idx)]
    if not (group_sets[0].isdisjoint(group_sets[1]) and
            group_sets[0].isdisjoint(group_sets[2]) and
            group_sets[1].isdisjoint(group_sets[2])):
        raise ValueError("Patient overlap detected across train/validation/test.")
    if len(np.unique(y[train_idx])) != 2:
        raise ValueError("Training split must contain both target classes.")

    print("Row/label/patient alignment verified.")
    print("Patient-disjoint train/validation/test split verified.")
    print(f"Rows: train={len(train_idx):,}, validation={len(val_idx):,}, test={len(test_idx):,}")
    print("Engineered feature checks:")
    print(f"  medication_changes_count range: {int(X_engineered['medication_changes_count'].min())}..{int(X_engineered['medication_changes_count'].max())}")
    print(f"  has_specialty counts: {X_engineered['has_specialty'].value_counts().sort_index().to_dict()}")
    print("Original medication columns, 'change', and 'diabetesMed' are retained.")

    experiment_specs = [
        {
            "name": "baseline",
            "X": X_baseline,
            "numeric_cols": base_numeric_cols,
            "categorical_cols": base_categorical_cols,
        },
        {
            "name": "engineered",
            "X": X_engineered,
            "numeric_cols": eng_numeric_cols,
            "categorical_cols": eng_categorical_cols,
        },
    ]
    comparison_rows = []
    threshold_rows = []
    metadata_variants = {}

    cv = GroupKFold(n_splits=5)
    for spec in experiment_specs:
        name = spec["name"]
        X = spec["X"]
        numeric_cols = spec["numeric_cols"]
        categorical_cols = spec["categorical_cols"]
        print(f"\n[{name.upper()}] Starting 5-fold GroupKFold CV on TRAINING rows only...")
        print("Preprocessing is refit inside every fold.")

        cv_model = make_pipeline(numeric_cols, categorical_cols)
        cv_prob = cross_val_predict(
            cv_model,
            X.iloc[train_idx],
            y[train_idx],
            groups=groups[train_idx],
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        cv_metrics, cv_pred, cv_cm = evaluate(
            y[train_idx], cv_prob, 0.5, name, "5-fold GroupKFold on training split"
        )
        cv_metrics["grouping"] = "patient_nbr"
        cv_metrics["preprocessing_refit_inside_each_fold"] = True
        cv_metrics["n_unique_patients"] = int(len(np.unique(groups[train_idx])))
        (out_dir / f"{name}_grouped_cv_metrics.json").write_text(
            json.dumps(cv_metrics, indent=2), encoding="utf-8"
        )
        pd.DataFrame({
            "source_row_index": train_idx,
            "y_true": y[train_idx],
            "y_pred_at_0_5": cv_pred,
            "probability_readmitted_30d": cv_prob,
        }).to_csv(out_dir / f"{name}_grouped_cv_predictions.csv", index=False)
        comparison_rows.append(cv_metrics)

        print(f"[{name.upper()}] Fit on training, evaluate on validation...")
        final_pipeline = make_pipeline(numeric_cols, categorical_cols)
        final_pipeline.fit(X.iloc[train_idx], y[train_idx])
        val_prob = final_pipeline.predict_proba(X.iloc[val_idx])[:, 1]
        val_metrics, val_pred, val_cm = evaluate(y[val_idx], val_prob, 0.5, name, "validation")
        (out_dir / f"{name}_validation_metrics.json").write_text(
            json.dumps(val_metrics, indent=2), encoding="utf-8"
        )
        pd.DataFrame({
            "source_row_index": val_idx,
            "y_true": y[val_idx],
            "y_pred_at_0_5": val_pred,
            "probability_readmitted_30d": val_prob,
        }).to_csv(out_dir / f"{name}_validation_predictions.csv", index=False)
        pd.DataFrame(
            classification_report(y[val_idx], val_pred, labels=[0, 1],
                                  output_dict=True, zero_division=0)
        ).transpose().to_csv(out_dir / f"{name}_validation_classification_report.csv")
        pd.DataFrame(
            val_cm,
            index=["actual_0_not_readmitted", "actual_1_readmitted"],
            columns=["predicted_0_not_readmitted", "predicted_1_readmitted"],
        ).to_csv(out_dir / f"{name}_validation_confusion_matrix.csv")
        joblib.dump(final_pipeline, out_dir / f"{name}_random_forest_pipeline.joblib")
        comparison_rows.append(val_metrics)

        # Threshold analysis is done on validation only and is explicitly marked
        # exploratory because the same validation set selects and evaluates it.
        best, sweep = best_f1_threshold(y[val_idx], val_prob)
        best_pred = (val_prob >= best["threshold"]).astype(np.int8)
        best_row = {
            "model_variant": name,
            "evaluation_split": "validation (threshold selected on same validation set; exploratory)",
            **best,
            "accuracy": float(accuracy_score(y[val_idx], best_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y[val_idx], best_pred)),
            "average_precision_pr_auc": float(average_precision_score(y[val_idx], val_prob)),
            "roc_auc": float(roc_auc_score(y[val_idx], val_prob)),
            "true_negatives": int(confusion_matrix(y[val_idx], best_pred, labels=[0, 1])[0, 0]),
            "false_positives": int(confusion_matrix(y[val_idx], best_pred, labels=[0, 1])[0, 1]),
            "false_negatives": int(confusion_matrix(y[val_idx], best_pred, labels=[0, 1])[1, 0]),
            "true_positives": int(confusion_matrix(y[val_idx], best_pred, labels=[0, 1])[1, 1]),
        }
        threshold_rows.append(best_row)
        sweep.insert(0, "model_variant", name)
        sweep.to_csv(out_dir / f"{name}_validation_threshold_sweep.csv", index=False)

        metadata_variants[name] = {
            "numeric_features": numeric_cols,
            "categorical_features": categorical_cols,
            "feature_count_after_encoding": int(
                final_pipeline.named_steps["preprocessor"].transform(X.iloc[train_idx[:min(2, len(train_idx))]]).shape[1]
            ),
            "model_parameters": final_pipeline.named_steps["model"].get_params(),
            "cv_pr_auc": cv_metrics["average_precision_pr_auc"],
            "cv_roc_auc": cv_metrics["roc_auc"],
            "validation_pr_auc": val_metrics["average_precision_pr_auc"],
            "validation_roc_auc": val_metrics["roc_auc"],
            "validation_f1_at_0_5": val_metrics["f1"],
        }

    pd.DataFrame(comparison_rows).to_csv(out_dir / "experiment_metrics_comparison.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(out_dir / "validation_best_f1_thresholds_exploratory.csv", index=False)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(raw_path),
        "split_assignments": str(assignments_path),
        "split_alignment_verified": True,
        "patient_disjoint_splits_verified": True,
        "cv": "5-fold GroupKFold by patient_nbr within training split",
        "preprocessing_refit_inside_each_cv_fold": True,
        "feature_engineering_function": "add_care_continuum_engineered_features",
        "medication_features_counted": MEDICATION_FEATURES_FOR_COUNT,
        "medication_changes_count_rule": "count exact Up/Down labels after trim and case-fold across 10 retained medication columns",
        "has_specialty_rule": "1 for nonblank, non-'?', non-Unknown medical_specialty_group; otherwise 0",
        "original_medication_columns_retained": True,
        "change_and_diabetesMed_untouched": True,
        "test_rows_loaded_for_split_integrity_only": int(len(test_idx)),
        "test_metrics_computed": False,
        "threshold_selection_note": "Best-F1 thresholds use validation predictions and are exploratory; they are not unbiased final performance estimates.",
        "variants": metadata_variants,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nEXPERIMENT COMPLETE. Key comparison:")
    comparison = pd.DataFrame(comparison_rows)
    print(comparison[[
        "model_variant", "evaluation_split", "average_precision_pr_auc", "roc_auc",
        "precision", "recall", "f1", "balanced_accuracy",
    ]].to_string(index=False))
    print("\nExploratory validation thresholds maximizing F1:")
    print(pd.DataFrame(threshold_rows)[[
        "model_variant", "threshold", "precision", "recall", "f1"
    ]].to_string(index=False))
    print(f"\nAll outputs saved to: {out_dir}")
    print("Test data was not used for fitting, tuning, or scoring.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
