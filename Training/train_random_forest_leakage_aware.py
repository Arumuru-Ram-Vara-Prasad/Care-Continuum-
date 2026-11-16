"""Leakage-aware Random Forest workflow for Care Continuum.

- Reuses the existing patient-disjoint train/validation/test assignments.
- Rebuilds raw predictor columns from the already-pruned CSV.
- Fits imputation, scaling, and one-hot encoding inside each CV fold via Pipeline.
- Runs 5-fold GroupKFold by patient_nbr on TRAINING rows only.
- Fits the final preprocessing+Random Forest pipeline on TRAINING rows only.
- Evaluates on VALIDATION at threshold 0.5.
- Does not score or tune on TEST.

Place this script in <project>/Training/ and run:
    python .\\Training\\train_random_forest_leakage_aware.py
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
    accuracy_score, average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "readmitted_30d"
GROUP_COL = "patient_nbr"
NOT_MEASURED_FEATURES = ["max_glu_serum", "A1Cresult"]
NUMERIC_FEATURES = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]
CATEGORICAL_CODE_FEATURES = [
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
]
DRUG_FEATURES_TO_EXCLUDE = [
    "chlorpropamide", "acetohexamide", "tolbutamide", "acarbose", "miglitol",
    "troglitazone", "tolazamide", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone", "examide", "citoglipton",
]


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


def clean_text_series(series):
    out = series.map(lambda x: np.nan if pd.isna(x) else str(x).strip())
    return out.replace({"": np.nan, "?": np.nan})


def load_raw_features(raw_path: Path):
    df = pd.read_csv(raw_path, dtype={GROUP_COL: "string"}, keep_default_na=False, low_memory=False)
    for required in [TARGET_COL, GROUP_COL, *NUMERIC_FEATURES, *CATEGORICAL_CODE_FEATURES]:
        if required not in df.columns:
            raise ValueError(f"Required column missing from pruned CSV: {required}")

    for col in NOT_MEASURED_FEATURES:
        if col in df.columns:
            text = df[col].astype("string").str.strip()
            mask = text.isna() | text.eq("") | text.str.casefold().eq("none")
            df.loc[mask.fillna(False), col] = "Not measured"

    for col in df.columns:
        if col not in NOT_MEASURED_FEATURES and (
            pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])
        ):
            df[col] = clean_text_series(df[col])

    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    if y.isna().any() or not set(y.unique()).issubset({0, 1}):
        raise ValueError(f"{TARGET_COL} must contain only complete binary 0/1 labels.")
    y = y.astype("int8").to_numpy()

    groups = df[GROUP_COL].astype("string").str.strip()
    if groups.isna().any() or groups.eq("").any():
        raise ValueError("Missing/blank patient_nbr; cannot do patient-grouped CV.")
    groups = groups.astype(str).to_numpy()

    # Keep only predictors: never pass target, patient ID, or encounter ID into model.
    excluded = [TARGET_COL, GROUP_COL, "encounter_id", *DRUG_FEATURES_TO_EXCLUDE]
    X = df.drop(columns=[c for c in excluded if c in df.columns]).copy()
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in CATEGORICAL_CODE_FEATURES:
        X[col] = clean_text_series(X[col])
    numeric_cols = list(NUMERIC_FEATURES)
    missing_numeric = [c for c in numeric_cols if c not in X.columns]
    if missing_numeric:
        raise ValueError(f"Numeric predictors missing after exclusions: {missing_numeric}")
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    for col in categorical_cols:
        X[col] = clean_text_series(X[col])
    return df, X, y, groups, numeric_cols, categorical_cols


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


def evaluate(y_true, probability, threshold, split_name):
    pred = (probability >= threshold).astype(np.int8)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    return {
        "model": "random_forest_with_fold_local_preprocessing",
        "evaluation_split": split_name,
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "average_precision_pr_auc": float(average_precision_score(y_true, probability)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "confusion_matrix_labels_0_then_1": cm.tolist(),
    }, pred, cm


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
    out_dir = root / "models" / "random_forest_leakage_aware"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Raw cleaned/pruned input: {raw_path}")
    print(f"Existing split assignments: {assignments_path}")
    raw_df, X, y, groups, numeric_cols, categorical_cols = load_raw_features(raw_path)
    assignments = pd.read_csv(assignments_path, dtype={GROUP_COL: "string"})
    required = {"source_row_index", GROUP_COL, TARGET_COL, "split"}
    if not required.issubset(assignments.columns):
        raise ValueError(f"split_assignments.csv missing required columns: {sorted(required - set(assignments.columns))}")
    if assignments["source_row_index"].duplicated().any():
        raise ValueError("source_row_index must be unique in split_assignments.csv.")
    if len(assignments) != len(raw_df):
        raise ValueError(
            f"Row-count mismatch: pruned CSV has {len(raw_df)} rows, split assignments have {len(assignments)}. "
            "Cannot safely reuse assignments; no model trained."
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
    if not (group_sets[0].isdisjoint(group_sets[1]) and group_sets[0].isdisjoint(group_sets[2]) and group_sets[1].isdisjoint(group_sets[2])):
        raise ValueError("Patient overlap detected across train/validation/test.")
    if len(np.unique(y[train_idx])) != 2:
        raise ValueError("Training split must contain both target classes.")

    print("Row/label/patient alignment verified.")
    print("Patient-disjoint train/validation/test split verified.")
    print(f"Rows: train={len(train_idx):,}, validation={len(val_idx):,}, test={len(test_idx):,}")
    print(f"Predictors: {X.shape[1]} ({len(numeric_cols)} numeric, {len(categorical_cols)} categorical)")
    print("\nStarting 5-fold GroupKFold CV on TRAINING rows only. Preprocessing is refit inside every fold.")

    cv_model = make_pipeline(numeric_cols, categorical_cols)
    cv = GroupKFold(n_splits=5)
    cv_prob = cross_val_predict(
        cv_model, X.iloc[train_idx], y[train_idx],
        groups=groups[train_idx], cv=cv, method="predict_proba", n_jobs=1
    )[:, 1]
    cv_metrics, cv_pred, cv_cm = evaluate(y[train_idx], cv_prob, 0.5, "5-fold GroupKFold on training split")
    cv_metrics["grouping"] = "patient_nbr"
    cv_metrics["preprocessing_refit_inside_each_fold"] = True
    cv_metrics["n_unique_patients"] = int(len(np.unique(groups[train_idx])))
    (out_dir / "grouped_cv_metrics.json").write_text(json.dumps(cv_metrics, indent=2), encoding="utf-8")
    pd.DataFrame({
        "source_row_index": train_idx,
        "y_true": y[train_idx],
        "y_pred": cv_pred,
        "probability_readmitted_30d": cv_prob,
    }).to_csv(out_dir / "grouped_cv_predictions.csv", index=False)

    print("\nFitting pipeline on all training rows, then evaluating on validation...")
    final_pipeline = make_pipeline(numeric_cols, categorical_cols)
    final_pipeline.fit(X.iloc[train_idx], y[train_idx])
    val_prob = final_pipeline.predict_proba(X.iloc[val_idx])[:, 1]
    val_metrics, val_pred, val_cm = evaluate(y[val_idx], val_prob, 0.5, "validation")
    joblib.dump(final_pipeline, out_dir / "random_forest_pipeline.joblib")
    (out_dir / "validation_metrics.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")
    pd.DataFrame({
        "source_row_index": val_idx,
        "y_true": y[val_idx],
        "y_pred": val_pred,
        "probability_readmitted_30d": val_prob,
    }).to_csv(out_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(classification_report(y[val_idx], val_pred, labels=[0, 1], output_dict=True, zero_division=0)).transpose().to_csv(
        out_dir / "validation_classification_report.csv"
    )
    pd.DataFrame(val_cm, index=["actual_0_not_readmitted", "actual_1_readmitted"],
                 columns=["predicted_0_not_readmitted", "predicted_1_readmitted"]).to_csv(
        out_dir / "validation_confusion_matrix.csv"
    )
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(raw_path),
        "split_assignments": str(assignments_path),
        "split_alignment_verified": True,
        "patient_disjoint_splits_verified": True,
        "cv": "5-fold GroupKFold by patient_nbr within training split",
        "preprocessing_refit_inside_each_cv_fold": True,
        "model_parameters": final_pipeline.named_steps["model"].get_params(),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "feature_count_after_encoding": int(final_pipeline.named_steps["preprocessor"].transform(X.iloc[train_idx[:min(2, len(train_idx))]]).shape[1]),
        "test_rows_loaded_for_integrity_only": int(len(test_idx)),
        "test_metrics_computed": False,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nGROUPED CV METRICS (training split only):")
    print(json.dumps(cv_metrics, indent=2))
    print("\nVALIDATION METRICS (threshold 0.5):")
    print(json.dumps(val_metrics, indent=2))
    print(f"\nAll outputs saved to: {out_dir}")
    print("Test set was not used for training, tuning, or scoring.")
    return 0


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
