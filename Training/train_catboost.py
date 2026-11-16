
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
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

try:
    from catboost import CatBoostClassifier
except ImportError as exc:
    raise SystemExit(
        "CatBoost is missing. Install it with:\n"
        "python -m pip install catboost"
    ) from exc


# ============================================================
# CARE CONTINUUM - CATBOOST EXPERIMENT
# Original feature set only
# Patient-grouped cross-validation
# Test set is never scored
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

# Exclusions already used in the existing preprocessing workflow.
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

# Do not use the features from the latest feature-engineering
# experiment in this baseline CatBoost comparison.
EXPERIMENTAL_FEATURES = [
    "medication_changes_count",
    "has_specialty",
]

PARAMS = {
    "iterations": 1000,
    "depth": 6,
    "learning_rate": 0.05,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "auto_class_weights": "Balanced",
    "random_seed": SEED,
    "verbose": False,
    "allow_writing_files": False,
    "thread_count": -1,
}


def find_project_root():
    script_path = Path(__file__).resolve()

    for path in [script_path.parent, *script_path.parents]:
        if (path / "dataset").is_dir():
            return path

    raise FileNotFoundError(
        "Cannot find the project root. Save the script in "
        "Training/ inside your Care Continuum project."
    )


def find_pruned_dataset(dataset_dir):
    files = sorted(
        dataset_dir.glob("diabetic_data_drugs_pruned*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not files:
        raise FileNotFoundError(
            f"No diabetic_data_drugs_pruned*.csv found in {dataset_dir}"
        )

    print(f"Dataset selected: {files[0].name}")
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
            print(f"Split file selected: {path}")
            return path

    raise FileNotFoundError(
        "split_assignments.csv was not found in the expected locations."
    )


def clean_text(series):
    """Treat blank strings and '?' as missing, preserving 'None'."""
    result = series.map(
        lambda value: np.nan if pd.isna(value)
        else str(value).strip()
    )
    return result.replace({"": np.nan, "?": np.nan})


def prepare_features(raw):
    df = raw.copy()

    if TARGET not in df.columns:
        raise KeyError(f"Missing target column: {TARGET}")

    if GROUP not in df.columns:
        raise KeyError(f"Missing patient grouping column: {GROUP}")

    # Preserve the existing semantic interpretation of lab results.
    for col in NOT_MEASURED_FEATURES:
        if col not in df.columns:
            continue

        values = df[col].map(
            lambda value: np.nan if pd.isna(value)
            else str(value).strip()
        )

        is_not_measured = (
            values.isna()
            | values.eq("")
            | values.str.casefold().eq("none")
        )

        values = values.mask(is_not_measured, "Not measured")
        values = values.mask(values.eq("?"), np.nan)
        df[col] = values

    # Keep the existing missing-value conventions for other columns.
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
            f"{TARGET} must contain only nonmissing 0/1 labels."
        )

    y = y.astype("int8").reset_index(drop=True)

    groups = df[GROUP].astype("string").str.strip()

    if groups.isna().any() or groups.eq("").any():
        raise ValueError("patient_nbr has missing or blank values.")

    groups = groups.astype(str).reset_index(drop=True)

    # IDs are used for alignment/grouping only, not model features.
    excluded = [
        TARGET,
        GROUP,
        "encounter_id",
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
            f"Expected numeric feature(s) missing: {missing_numeric}"
        )

    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    categorical_features = [
        c for c in X.columns if c not in NUMERIC_FEATURES
    ]

    # CatBoost accepts categorical values as strings.
    # Numeric missing values remain NaN; CatBoost handles those natively.
    for col in categorical_features:
        X[col] = clean_text(X[col]).fillna("Unknown").astype(str)

    X = X.reset_index(drop=True)

    if X.empty or len(X) != len(y):
        raise ValueError("Feature matrix and target are not aligned.")

    print(f"Feature count: {X.shape[1]}")
    print(f"Numeric features: {len(NUMERIC_FEATURES)}")
    print(f"Categorical features: {len(categorical_features)}")
    print("Experimental features are excluded.")

    return X, y, groups, categorical_features


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
            f"Split assignments must contain: {sorted(required)}"
        )

    assignments["source_row_index"] = pd.to_numeric(
        assignments["source_row_index"],
        errors="coerce",
    )

    if assignments["source_row_index"].isna().any():
        raise ValueError("Invalid source_row_index in split file.")

    assignments = assignments.sort_values(
        "source_row_index"
    ).reset_index(drop=True)

    expected_indices = np.arange(len(raw))

    if (
        len(assignments) != len(raw)
        or not np.array_equal(
            assignments["source_row_index"].astype(int).to_numpy(),
            expected_indices,
        )
    ):
        raise ValueError(
            "Split assignments do not align with the source CSV. "
            "Stop rather than assuming row alignment."
        )

    split_y = pd.to_numeric(assignments[TARGET], errors="coerce")

    if split_y.isna().any() or not np.array_equal(
        split_y.astype(int).to_numpy(),
        y.to_numpy(),
    ):
        raise ValueError(
            "Target labels in split_assignments.csv do not match the source."
        )

    split_groups = (
        assignments[GROUP].astype("string").str.strip()
    )

    if not np.array_equal(
        split_groups.astype(str).to_numpy(),
        groups.astype(str).to_numpy(),
    ):
        raise ValueError(
            "Patient IDs in split assignments do not match the source CSV."
        )

    if set(assignments["split"].astype(str)) != {
        "train", "validation", "test"
    }:
        raise ValueError("Unexpected labels in the split column.")

    group_split_counts = (
        assignments.assign(
            _patient=split_groups.astype(str)
        )
        .groupby("_patient")["split"]
        .nunique()
    )

    if (group_split_counts > 1).any():
        raise ValueError(
            "Patient leakage detected across train/validation/test."
        )

    print("Source-row alignment verified.")
    print("Target alignment verified.")
    print("Patient-disjoint splits verified.")

    return assignments


def calculate_metrics(y_true, probabilities, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "average_precision_pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix_labels_0_then_1": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
    }


def make_threshold_sweep(y_true, probabilities, variant):
    rows = []

    for threshold in np.round(np.arange(0.01, 0.991, 0.005), 3):
        result = calculate_metrics(
            y_true,
            probabilities,
            threshold=float(threshold),
        )

        rows.append({
            "model_variant": variant,
            "threshold": float(threshold),
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "balanced_accuracy": result["balanced_accuracy"],
        })

    return pd.DataFrame(rows)


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2),
        encoding="utf-8",
    )


def main():
    root = find_project_root()
    output_dir = root / "models" / "catboost"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = find_pruned_dataset(root / "dataset")
    split_path = find_split_assignments(root)

    raw = pd.read_csv(
        dataset_path,
        dtype={GROUP: "string"},
        keep_default_na=False,
        low_memory=False,
    )

    X, y, groups, categorical_features = prepare_features(raw)

    assignments = verify_splits(
        raw, y, groups, split_path
    )

    split_names = assignments["split"].astype(str).to_numpy()

    train_idx = np.flatnonzero(split_names == "train")
    validation_idx = np.flatnonzero(split_names == "validation")
    test_idx = np.flatnonzero(split_names == "test")

    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    groups_train = groups.iloc[train_idx].reset_index(drop=True)

    X_validation = X.iloc[validation_idx].reset_index(drop=True)
    y_validation = y.iloc[validation_idx].reset_index(drop=True)

    print(
        f"Rows: train={len(train_idx):,}, "
        f"validation={len(validation_idx):,}, "
        f"test={len(test_idx):,}"
    )
    print("Test rows are not used for model evaluation.")

    # --------------------------------------------------------
    # 1. Five-fold GroupKFold on training encounters
    # CatBoost uses categorical features natively. No globally
    # fitted imputer, scaler or encoder is used before CV.
    # --------------------------------------------------------

    cv = GroupKFold(n_splits=N_SPLITS)
    oof_probabilities = np.full(len(X_train), np.nan)
    fold_results = []

    for fold, (fit_idx, held_idx) in enumerate(
        cv.split(X_train, y_train, groups=groups_train),
        start=1,
    ):
        print(f"Training fold {fold}/{N_SPLITS}...")

        model = CatBoostClassifier(**PARAMS)

        model.fit(
            X_train.iloc[fit_idx],
            y_train.iloc[fit_idx],
            cat_features=categorical_features,
            verbose=False,
        )

        probabilities = model.predict_proba(
            X_train.iloc[held_idx]
        )[:, 1]

        oof_probabilities[held_idx] = probabilities

        result = calculate_metrics(
            y_train.iloc[held_idx],
            probabilities,
        )

        result["fold"] = fold
        result["training_rows"] = int(len(fit_idx))
        result["heldout_rows"] = int(len(held_idx))
        fold_results.append(result)

    if np.isnan(oof_probabilities).any():
        raise RuntimeError("Some training rows lack out-of-fold predictions.")

    cv_metrics = calculate_metrics(y_train, oof_probabilities)

    cv_metrics.update({
        "model": "CatBoostClassifier",
        "evaluation_split": "5-fold GroupKFold on training split",
        "grouping": GROUP,
        "preprocessing_refit_inside_each_fold": True,
        "categorical_handling": "native CatBoost categorical features",
        "n_unique_patients": int(groups_train.nunique()),
        "fold_metrics": fold_results,
    })

    write_json(
        output_dir / "grouped_cv_metrics.json",
        cv_metrics,
    )

    pd.DataFrame({
        "source_row_index": train_idx,
        TARGET: y_train,
        "readmission_probability": oof_probabilities,
    }).to_csv(
        output_dir / "grouped_cv_predictions.csv",
        index=False,
    )

    pd.DataFrame(fold_results).to_csv(
        output_dir / "grouped_cv_fold_metrics.csv",
        index=False,
    )

    make_threshold_sweep(
        y_train,
        oof_probabilities,
        "catboost_grouped_cv_oof",
    ).to_csv(
        output_dir / "grouped_cv_threshold_sweep.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 2. Fit the final CatBoost variant on training only
    # 3. Evaluate once on validation
    # --------------------------------------------------------

    print("Fitting CatBoost on the complete training split...")

    final_model = CatBoostClassifier(**PARAMS)

    final_model.fit(
        X_train,
        y_train,
        cat_features=categorical_features,
        verbose=False,
    )

    final_model.save_model(
        str(output_dir / "catboost_model.cbm")
    )

    probabilities = final_model.predict_proba(
        X_validation
    )[:, 1]

    val_metrics = calculate_metrics(
        y_validation,
        probabilities,
        threshold=0.5,
    )

    val_metrics.update({
        "model": "CatBoostClassifier",
        "evaluation_split": "validation",
        "test_metrics_computed": False,
    })

    write_json(
        output_dir / "validation_metrics.json",
        val_metrics,
    )

    predictions = (probabilities >= 0.5).astype(int)

    pd.DataFrame({
        "source_row_index": validation_idx,
        TARGET: y_validation,
        "readmission_probability": probabilities,
        "prediction_at_0_5": predictions,
    }).to_csv(
        output_dir / "validation_predictions.csv",
        index=False,
    )

    pd.DataFrame(
        confusion_matrix(
            y_validation,
            predictions,
            labels=[0, 1],
        ),
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    ).to_csv(
        output_dir / "validation_confusion_matrix.csv"
    )

    pd.DataFrame(
        classification_report(
            y_validation,
            predictions,
            labels=[0, 1],
            output_dict=True,
            zero_division=0,
        )
    ).transpose().to_csv(
        output_dir / "validation_classification_report.csv"
    )

    sweep = make_threshold_sweep(
        y_validation,
        probabilities,
        "catboost_validation",
    )

    sweep.to_csv(
        output_dir / "validation_threshold_sweep.csv",
        index=False,
    )

    best = sweep.loc[sweep["f1"].idxmax()]

    write_json(
        output_dir / "validation_best_f1_threshold_exploratory.json",
        {
            "warning": (
                "Exploratory only: threshold selection and evaluation "
                "use the same validation set."
            ),
            "threshold": float(best["threshold"]),
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "f1": float(best["f1"]),
            "balanced_accuracy": float(best["balanced_accuracy"]),
        },
    )

    importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": final_model.get_feature_importance(),
    }).sort_values("importance", ascending=False)

    importance.to_csv(
        output_dir / "feature_importance.csv",
        index=False,
    )

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "CatBoostClassifier",
        "dataset": str(dataset_path),
        "split_assignments": str(split_path),
        "patient_disjoint_splits_verified": True,
        "source_row_alignment_verified": True,
        "cv": "5-fold GroupKFold by patient_nbr",
        "parameters": PARAMS,
        "original_feature_set_used": True,
        "experimental_features_excluded": EXPERIMENTAL_FEATURES,
        "test_rows_loaded_for_integrity_only": int(len(test_idx)),
        "test_metrics_computed": False,
        "design_caveat": (
            "discharge_disposition_id is currently included. If predictions "
            "are intended at admission, it must be removed because it is "
            "not available at that prediction time."
        ),
    }

    write_json(output_dir / "run_metadata.json", metadata)

    print("\n=== CATBOOST RESULTS ===")

    print("\nGrouped 5-fold CV (out-of-fold):")
    for key in [
        "accuracy", "balanced_accuracy", "precision",
        "recall", "f1", "average_precision_pr_auc", "roc_auc",
    ]:
        print(f"{key}: {cv_metrics[key]:.5f}")

    print("\nValidation at threshold 0.5:")
    for key in [
        "accuracy", "balanced_accuracy", "precision",
        "recall", "f1", "average_precision_pr_auc", "roc_auc",
    ]:
        print(f"{key}: {val_metrics[key]:.5f}")

    print("\nExploratory validation F1-max threshold:")
    print(
        f"threshold={best['threshold']:.3f}, "
        f"precision={best['precision']:.4f}, "
        f"recall={best['recall']:.4f}, "
        f"F1={best['f1']:.4f}"
    )

    print(f"\nOutputs saved to: {output_dir}")
    print("The test set was not scored.")


if __name__ == "__main__":
    main()
