"""Train a class-weighted Logistic Regression experiment for Care Continuum.

Expected files (either in dataset/ or dataset/model_ready/):
  X_train.npz, X_validation.npz, X_test.npz
  y_train.csv, y_validation.csv, y_test.csv

Fits only on the training split. Evaluates on validation at threshold 0.5.
The test split is loaded only for integrity/shape checks; it is not scored.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
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

FILES = [
    "X_train.npz", "X_validation.npz", "X_test.npz",
    "y_train.csv", "y_validation.csv", "y_test.csv",
]
TARGET_CANDIDATES = ("readmitted_30d", "target", "y")


def find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "dataset").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find the project root. Put this script in a subfolder of your project, "
        "where a 'dataset' folder exists."
    )


def find_data_dir(project_root: Path) -> Path:
    candidates = [project_root / "dataset" / "model_ready", project_root / "dataset"]
    for folder in candidates:
        if folder.is_dir() and all((folder / name).is_file() for name in FILES):
            return folder
    details = "\n".join(f"  - {folder}" for folder in candidates)
    raise FileNotFoundError(
        "Could not find all six expected split files in either location:\n"
        f"{details}\nExpected: {', '.join(FILES)}"
    )


def load_target(path: Path) -> tuple[np.ndarray, pd.DataFrame, str]:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Target file is empty: {path}")
    target_col = next((c for c in TARGET_CANDIDATES if c in frame.columns), None)
    if target_col is None:
        non_index_cols = [
            c for c in frame.columns
            if not c.lower().startswith("unnamed:")
            and c.lower() not in {"index", "row_index", "source_row_index"}
        ]
        if len(non_index_cols) == 1:
            target_col = non_index_cols[0]
        else:
            raise ValueError(
                f"Could not identify the target column in {path.name}. "
                f"Columns found: {list(frame.columns)}"
            )
    values = pd.to_numeric(frame[target_col], errors="raise").to_numpy()
    if pd.isna(values).any():
        raise ValueError(f"Missing target values found in {path.name}")
    unique = set(np.unique(values).tolist())
    if not unique or not unique.issubset({0, 1}):
        raise ValueError(f"Expected a binary 0/1 target in {path.name}; found {sorted(unique)}")
    return values.astype(np.int8), frame, target_col


def load_matrix(path: Path):
    matrix = sparse.load_npz(path).tocsr()
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"Empty feature matrix: {path}")
    if not np.isfinite(matrix.data).all():
        raise ValueError(f"Non-finite feature values found in {path.name}")
    return matrix


def extract_source_indices(frame: pd.DataFrame):
    for col in ("source_row_index", "row_index", "original_index", "index"):
        if col in frame.columns:
            vals = frame[col].dropna()
            if len(vals) == len(frame):
                return set(vals.astype(str).tolist())
    for col in frame.columns:
        if col.lower().startswith("unnamed:"):
            vals = frame[col].dropna()
            if len(vals) == len(frame):
                return set(vals.astype(str).tolist())
    return None


def main() -> int:
    project_root = find_project_root()
    data_dir = find_data_dir(project_root)
    output_dir = project_root / "models" / "logistic_regression_weighted"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {project_root}")
    print(f"Data directory: {data_dir}")
    print("Loading existing train, validation, and test splits...")

    X_train = load_matrix(data_dir / "X_train.npz")
    X_val = load_matrix(data_dir / "X_validation.npz")
    X_test = load_matrix(data_dir / "X_test.npz")
    y_train, train_frame, train_target_col = load_target(data_dir / "y_train.csv")
    y_val, val_frame, val_target_col = load_target(data_dir / "y_validation.csv")
    y_test, test_frame, test_target_col = load_target(data_dir / "y_test.csv")

    for name, X, y in (("train", X_train, y_train), ("validation", X_val, y_val), ("test", X_test, y_test)):
        if X.shape[0] != len(y):
            raise ValueError(f"{name}: X has {X.shape[0]} rows but y has {len(y)} rows")
    if len({X_train.shape[1], X_val.shape[1], X_test.shape[1]}) != 1:
        raise ValueError("Train, validation, and test matrices have different feature counts")
    if len(np.unique(y_train)) != 2:
        raise ValueError("Training target must contain both classes (0 and 1)")

    # Check source-row indices if the target CSVs preserve them.
    index_sets = {
        "train": extract_source_indices(train_frame),
        "validation": extract_source_indices(val_frame),
        "test": extract_source_indices(test_frame),
    }
    if all(index_sets[n] is not None for n in ("train", "validation", "test")):
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
            overlap = index_sets[a] & index_sets[b]
            if overlap:
                raise ValueError(f"Source-row indices overlap between {a} and {b}: {len(overlap)} rows")
        print("Source-row index check: no overlap between splits.")
    else:
        print("Note: source-row indices were not present in all target CSVs; overlap check skipped.")

    print(f"Rows: train={X_train.shape[0]:,}, validation={X_val.shape[0]:,}, test={X_test.shape[0]:,}")
    print(f"Encoded feature count: {X_train.shape[1]:,}")
    print(f"Training positive rate: {float(y_train.mean()):.2%}")
    print("Training class-weighted Logistic Regression (class_weight='balanced')...")

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Validation only; test split is not scored or used for tuning.
    val_prob = model.predict_proba(X_val)[:, 1]
    threshold = 0.5
    val_pred = (val_prob >= threshold).astype(np.int8)
    try:
        roc_auc = float(roc_auc_score(y_val, val_prob))
    except ValueError:
        roc_auc = None

    cm = confusion_matrix(y_val, val_pred, labels=[0, 1])
    metrics = {
        "model": "class_weighted_logistic_regression",
        "class_weight": "balanced",
        "evaluation_split": "validation",
        "threshold": threshold,
        "n_validation": int(len(y_val)),
        "positive_rate_validation": float(y_val.mean()),
        "accuracy": float(accuracy_score(y_val, val_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_val, val_pred)),
        "precision": float(precision_score(y_val, val_pred, zero_division=0)),
        "recall": float(recall_score(y_val, val_pred, zero_division=0)),
        "f1": float(f1_score(y_val, val_pred, zero_division=0)),
        "average_precision_pr_auc": float(average_precision_score(y_val, val_prob)),
        "roc_auc": roc_auc,
        "confusion_matrix_labels_0_then_1": cm.tolist(),
    }

    model_path = output_dir / "logistic_regression_weighted.joblib"
    predictions_path = output_dir / "validation_predictions.csv"
    metrics_path = output_dir / "validation_metrics.json"
    report_path = output_dir / "validation_classification_report.csv"
    metadata_path = output_dir / "run_metadata.json"
    cm_path = output_dir / "confusion_matrix.csv"

    joblib.dump(model, model_path)
    pd.DataFrame({
        "y_true": y_val,
        "y_pred": val_pred,
        "probability_readmitted_30d": val_prob,
    }).to_csv(predictions_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(
        classification_report(y_val, val_pred, labels=[0, 1], output_dict=True, zero_division=0)
    ).transpose().to_csv(report_path, index=True)
    pd.DataFrame(
        cm,
        index=["actual_0_not_readmitted", "actual_1_readmitted"],
        columns=["predicted_0_not_readmitted", "predicted_1_readmitted"],
    ).to_csv(cm_path)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "sklearn.linear_model.LogisticRegression",
        "parameters": model.get_params(),
        "training_data_dir": str(data_dir),
        "feature_count": int(X_train.shape[1]),
        "train_rows": int(X_train.shape[0]),
        "validation_rows": int(X_val.shape[0]),
        "test_rows_loaded_for_integrity_only": int(X_test.shape[0]),
        "target_columns": {
            "train": train_target_col,
            "validation": val_target_col,
            "test": test_target_col,
        },
        "test_metrics_computed": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nCLASS-WEIGHTED LOGISTIC REGRESSION — VALIDATION METRICS (threshold = 0.5)")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("\nSaved files:")
    for path in (model_path, predictions_path, metrics_path, report_path, cm_path, metadata_path):
        print(f"  {path}")
    print("\nDone. The test set was not used for fitting, threshold tuning, or scoring.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
