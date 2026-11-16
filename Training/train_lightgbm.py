
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
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
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lightgbm import LGBMClassifier


# ============================================================
# CARE CONTINUUM - LIGHTGBM EXPERIMENT
# Original feature set; patient-grouped evaluation
# Test set is NOT scored
# ============================================================

TARGET = "readmitted_30d"
GROUP = "patient_nbr"
SEED = 42
N_SPLITS = 5

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

NOT_MEASURED_FEATURES = ["max_glu_serum", "A1Cresult"]

DRUG_COLUMNS_TO_EXCLUDE = [
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

EXPERIMENTAL_FEATURES = [
    "medication_changes_count",
    "has_specialty",
]

THRESHOLDS = np.round(np.arange(0.01, 0.991, 0.005), 3)

MODEL_PARAMS = {
    "objective": "binary",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": -1,
}


def find_project_root():
    script = Path(__file__).resolve()

    for path in [script.parent, *script.parents]:
        if (path / "dataset").is_dir():
            return path

    raise FileNotFoundError(
        "Project root not found. Save this script inside Training/, "
        "with dataset/ in the project root."
    )


def find_pruned_dataset(dataset_dir):
    files = sorted(
        dataset_dir.glob("diabetic_data_drugs_pruned*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not files:
        raise FileNotFoundError(
            f"No pruned CSV found in {dataset_dir}"
        )

    print(f"Using dataset: {files[0].name}")
    return files[0]


def find_split_assignments(root):
    candidates = [
        root / "dataset" / "split_assignments.csv",
        root / "dataset" / "model_ready" / "split_assignments.csv",
        root / "Preprocessing" / "split_assignments.csv",
        root / "split_assignments.csv",
    ]

    for path in candidates:
        if path.exists():
            print(f"Using split assignments: {path}")
            return path

    raise FileNotFoundError("split_assignments.csv not found.")


def clean_text(series):
    values = series.map(
        lambda v: np.nan if pd.isna(v) else str(v).strip()
    )
    return values.replace({"": np.nan, "?": np.nan})


def prepare_features(raw):
    df = raw.copy()

    for col in NOT_MEASURED_FEATURES:
        if col not in df.columns:
            continue

        values = df[col].map(
            lambda v: np.nan if pd.isna(v) else str(v).strip()
        )

        not_measured = (
            values.isna()
            | values.eq("")
            | values.str.casefold().eq("none")
        )

        values = values.mask(not_measured, "Not measured")
        values = values.mask(values.eq("?"), np.nan)
        df[col] = values

    for col in df.columns:
        if col in NOT_MEASURED_FEATURES:
            continue

        if (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
        ):
            df[col] = clean_text(df[col])

    y = pd.to_numeric(df[TARGET], errors="coerce")

    if y.isna().any() or not set(y.unique()).issubset({0, 1}):
        raise ValueError(
            f"{TARGET} must contain complete binary 0/1 labels."
        )

    y = y.astype("int8").reset_index(drop=True)

    groups = df[GROUP].astype("string").str.strip()

    if groups.isna().any() or groups.eq("").any():
        raise ValueError("patient_nbr has missing or blank values.")

    groups = groups.astype(str).reset_index(drop=True)

    # IDs, target, source target label, and experimental features
    # must not be used as model inputs.
    excluded = [
        TARGET,
        GROUP,
        "encounter_id",
        "readmitted",
        *DRUG_COLUMNS_TO_EXCLUDE,
        *EXPERIMENTAL_FEATURES,
    ]

    X = df.drop(
        columns=[c for c in excluded if c in df.columns]
    ).copy()

    missing_numeric = [
        c for c in NUMERIC_FEATURES if c not in X.columns
    ]

    if missing_numeric:
        raise KeyError(
            f"Expected numeric features missing: {missing_numeric}"
        )

    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    numeric_cols = list(NUMERIC_FEATURES)
    categorical_cols = [
        c for c in X.columns if c not in numeric_cols
    ]

    for col in categorical_cols:
        X[col] = clean_text(X[col])

    return (
        X.reset_index(drop=True),
        y,
        groups,
        numeric_cols,
        categorical_cols,
    )


def verify_splits(raw, y, groups, split_path):
    assignments = pd.read_csv(
        split_path,
        dtype={GROUP: "string"},
        keep_default_na=False,
    )

    required = {
        "source_row_index",
        GROUP,
        TARGET,
        "split",
    }

    if not required.issubset(assignments.columns):
        raise ValueError(
            f"Split file must contain {sorted(required)}"
        )

    assignments["source_row_index"] = pd.to_numeric(
        assignments["source_row_index"],
        errors="coerce",
    )

    if assignments["source_row_index"].isna().any():
        raise ValueError("Invalid source_row_index values.")

    assignments = assignments.sort_values(
        "source_row_index"
    ).reset_index(drop=True)

    expected = np.arange(len(raw))

    if (
        len(assignments) != len(raw)
        or not np.array_equal(
            assignments["source_row_index"].astype(int).to_numpy(),
            expected,
        )
    ):
        raise ValueError(
            "Split assignments do not align with the selected CSV."
        )

    split_y = pd.to_numeric(
        assignments[TARGET], errors="coerce"
    )

    if split_y.isna().any() or not np.array_equal(
        split_y.astype(int).to_numpy(), y.to_numpy()
    ):
        raise ValueError("Target labels do not match the source CSV.")

    split_groups = (
        assignments[GROUP].astype("string").str.strip()
    )

    if not np.array_equal(
        split_groups.astype(str).to_numpy(),
        groups.astype(str).to_numpy(),
    ):
        raise ValueError("Patient IDs do not match the source CSV.")

    if set(assignments["split"].astype(str)) != {
        "train", "validation", "test"
    }:
        raise ValueError("Unexpected split labels.")

    patient_split_counts = (
        assignments.assign(_patient=split_groups.astype(str))
        .groupby("_patient")["split"]
        .nunique()
    )

    if (patient_split_counts > 1).any():
        raise ValueError("Patient leakage detected across splits.")

    print("Row alignment verified.")
    print("Target alignment verified.")
    print("Patient-disjoint splits verified.")

    return assignments


def make_pipeline(numeric_cols, categorical_cols):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="constant",
            fill_value="Unknown",
        )),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ], remainder="drop", sparse_threshold=1.0)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", LGBMClassifier(**MODEL_PARAMS)),
    ])


def evaluate(y_true, probabilities, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = (probabilities >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predicted)
        ),
        "precision": float(
            precision_score(y_true, predicted, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predicted, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "average_precision_pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix_labels_0_then_1": confusion_matrix(
            y_true, predicted, labels=[0, 1]
        ).tolist(),
    }


def make_threshold_sweep(y_true, probabilities, name):
    rows = []

    for threshold in THRESHOLDS:
        result = evaluate(
            y_true, probabilities, float(threshold)
        )

        rows.append({
            "model_variant": name,
            "threshold": float(threshold),
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "balanced_accuracy": result["balanced_accuracy"],
        })

    return pd.DataFrame(rows)


def save_json(path, data):
    path.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )


def main():
    root = find_project_root()
    output = root / "models" / "lightgbm"
    output.mkdir(parents=True, exist_ok=True)

    dataset_path = find_pruned_dataset(root / "dataset")
    splits_path = find_split_assignments(root)

    raw = pd.read_csv(
        dataset_path,
        dtype={GROUP: "string"},
        keep_default_na=False,
        low_memory=False,
    )

    X, y, groups, numeric_cols, categorical_cols = (
        prepare_features(raw)
    )

    assignments = verify_splits(
        raw, y, groups, splits_path
    )

    split_names = assignments["split"].astype(str).to_numpy()

    train_rows = np.flatnonzero(split_names == "train")
    val_rows = np.flatnonzero(split_names == "validation")
    test_rows = np.flatnonzero(split_names == "test")

    X_train = X.iloc[train_rows].reset_index(drop=True)
    y_train = y.iloc[train_rows].reset_index(drop=True)
    groups_train = groups.iloc[train_rows].reset_index(drop=True)

    X_val = X.iloc[val_rows].reset_index(drop=True)
    y_val = y.iloc[val_rows].reset_index(drop=True)

    print(
        f"Training: {len(train_rows):,}; "
        f"validation: {len(val_rows):,}; "
        f"test: {len(test_rows):,}"
    )
    print(f"Total model inputs: {X.shape[1]}")
    print("Test set will not be scored.")

    # --------------------------------------------------------
    # Five-fold GroupKFold with a newly fitted preprocessing
    # pipeline in every fold.
    # --------------------------------------------------------

    cv = GroupKFold(n_splits=N_SPLITS)
    oof = np.full(len(X_train), np.nan)
    fold_results = []

    for fold, (fit_idx, held_idx) in enumerate(
        cv.split(X_train, y_train, groups=groups_train),
        start=1,
    ):
        print(f"Training fold {fold}/{N_SPLITS}...")

        pipeline = make_pipeline(numeric_cols, categorical_cols)
        pipeline.fit(
            X_train.iloc[fit_idx],
            y_train.iloc[fit_idx],
        )

        probability = pipeline.predict_proba(
            X_train.iloc[held_idx]
        )[:, 1]

        oof[held_idx] = probability

        row = evaluate(
            y_train.iloc[held_idx], probability
        )
        row["fold"] = fold
        fold_results.append(row)

    if np.isnan(oof).any():
        raise RuntimeError("Missing out-of-fold predictions.")

    cv_metrics = evaluate(y_train, oof)
    cv_metrics.update({
        "model": "LightGBM",
        "evaluation_split": "5-fold GroupKFold",
        "grouping": GROUP,
        "preprocessing_refit_inside_each_fold": True,
        "fold_metrics": fold_results,
    })

    save_json(output / "grouped_cv_metrics.json", cv_metrics)

    pd.DataFrame({
        "source_row_index": train_rows,
        TARGET: y_train,
        "readmission_probability": oof,
        "prediction_at_0_5": (oof >= 0.5).astype(int),
    }).to_csv(output / "grouped_cv_predictions.csv", index=False)

    pd.DataFrame(fold_results).to_csv(
        output / "grouped_cv_fold_metrics.csv", index=False
    )

    make_threshold_sweep(
        y_train, oof, "lightgbm_grouped_cv_oof"
    ).to_csv(output / "grouped_cv_threshold_sweep.csv", index=False)

    # Fit on all training rows; validation is evaluated separately.
    print("Fitting final LightGBM pipeline on training data...")

    final_pipeline = make_pipeline(numeric_cols, categorical_cols)
    final_pipeline.fit(X_train, y_train)

    joblib.dump(
        final_pipeline, output / "lightgbm_pipeline.joblib"
    )

    val_probability = final_pipeline.predict_proba(X_val)[:, 1]
    val_metrics = evaluate(y_val, val_probability)
    val_metrics.update({
        "model": "LightGBM",
        "evaluation_split": "validation",
        "test_metrics_computed": False,
    })

    save_json(output / "validation_metrics.json", val_metrics)

    val_prediction = (val_probability >= 0.5).astype(int)

    pd.DataFrame({
        "source_row_index": val_rows,
        TARGET: y_val,
        "readmission_probability": val_probability,
        "prediction_at_0_5": val_prediction,
    }).to_csv(output / "validation_predictions.csv", index=False)

    pd.DataFrame(
        confusion_matrix(y_val, val_prediction, labels=[0, 1]),
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    ).to_csv(output / "validation_confusion_matrix.csv")

    pd.DataFrame(
        classification_report(
            y_val,
            val_prediction,
            labels=[0, 1],
            output_dict=True,
            zero_division=0,
        )
    ).transpose().to_csv(
        output / "validation_classification_report.csv"
    )

    sweep = make_threshold_sweep(
        y_val, val_probability, "lightgbm_validation"
    )
    sweep.to_csv(output / "validation_threshold_sweep.csv", index=False)

    best = sweep.loc[sweep["f1"].idxmax()]

    save_json(
        output / "validation_best_f1_threshold_exploratory.json",
        {
            "warning": (
                "Exploratory only. The threshold was selected and "
                "evaluated on the same validation set."
            ),
            "threshold": float(best["threshold"]),
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "f1": float(best["f1"]),
            "balanced_accuracy": float(best["balanced_accuracy"]),
        },
    )

    feature_names = (
        final_pipeline.named_steps["preprocessor"]
        .get_feature_names_out()
    )
    importances = pd.DataFrame({
        "feature": feature_names,
        "importance": (
            final_pipeline.named_steps["model"].feature_importances_
        ),
    }).sort_values("importance", ascending=False)

    importances.to_csv(
        output / "feature_importance_encoded.csv", index=False
    )

    save_json(output / "run_metadata.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "LGBMClassifier",
        "input_csv": str(dataset_path),
        "split_assignments": str(splits_path),
        "patient_disjoint_splits_verified": True,
        "source_row_alignment_verified": True,
        "cv": "5-fold GroupKFold by patient_nbr",
        "preprocessing_refit_inside_each_cv_fold": True,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "original_feature_set_used": True,
        "experimental_features_excluded": EXPERIMENTAL_FEATURES,
        "model_parameters": MODEL_PARAMS,
        "test_rows_loaded_for_integrity_only": len(test_rows),
        "test_metrics_computed": False,
        "prediction_time": "discharge",
    })

    print("\n=== LIGHTGBM RESULTS ===")

    print("\nGrouped CV (out-of-fold):")
    for key in [
        "accuracy", "balanced_accuracy", "precision", "recall",
        "f1", "average_precision_pr_auc", "roc_auc",
    ]:
        print(f"{key}: {cv_metrics[key]:.5f}")

    print("\nValidation at threshold 0.5:")
    for key in [
        "accuracy", "balanced_accuracy", "precision", "recall",
        "f1", "average_precision_pr_auc", "roc_auc",
    ]:
        print(f"{key}: {val_metrics[key]:.5f}")

    print("\nExploratory validation F1-max threshold:")
    print(
        f"threshold={best['threshold']:.3f}, "
        f"precision={best['precision']:.4f}, "
        f"recall={best['recall']:.4f}, "
        f"F1={best['f1']:.4f}"
    )

    print(f"\nOutputs saved to: {output}")
    print("Test-set metrics were NOT computed.")


if __name__ == "__main__":
    main()
