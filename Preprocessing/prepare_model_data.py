from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# CARE CONTINUUM - VERIFIED PREPROCESSING SCRIPT
# Input: dataset/diabetic_data_drugs_pruned*.csv
# Run from the project's Preprocessing folder or project root.
# This script does not train a model or overwrite the source CSV.
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"
OUTPUT_DIR = DATASET_DIR / "model_ready"

TARGET_COL = "readmitted_30d"
GROUP_COL = "patient_nbr"
RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_FRACTION_OF_TRAINVAL = 0.1875

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

NOT_MEASURED_FEATURES = ["max_glu_serum", "A1Cresult"]

# 11 near-constant features verified in the separate drug analysis,
# plus the two constant drug columns removed in the earlier cleaning step.
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


def find_pruned_input():
    """Use only the already-pruned dataset; don't silently use an older file."""
    matches = list(DATASET_DIR.glob("diabetic_data_drugs_pruned*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find diabetic_data_drugs_pruned*.csv in {DATASET_DIR}.\n"
            "Place your existing pruned CSV in the project's dataset folder."
        )
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    print(f"Input selected: {matches[0].name}")
    return matches[0]


def clean_text_series(series):
    """Trim categorical text; mark blank strings and '?' as missing."""
    out = series.map(lambda x: np.nan if pd.isna(x) else str(x).strip())
    return out.replace({"": np.nan, "?": np.nan})


def make_input_audit(df):
    rows = []
    for col in df.columns:
        text = df[col].astype("string").str.strip()
        rows.append({
            "feature": col,
            "dtype_on_read": str(df[col].dtype),
            "unique_values_including_blank": int(df[col].nunique(dropna=False)),
            "blank_count_in_input": int(text.eq("").fillna(False).sum()),
            "question_mark_count_in_input": int(text.eq("?").fillna(False).sum()),
            "literal_None_count_in_input": int(
                text.str.casefold().eq("none").fillna(False).sum()
            ),
        })
    return pd.DataFrame(rows)


def make_feature_audit(X, numeric_cols):
    rows = []
    for col in X.columns:
        rows.append({
            "feature": col,
            "feature_type": "numeric" if col in numeric_cols else "categorical",
            "missing_count_after_semantic_recoding": int(X[col].isna().sum()),
            "missing_pct_after_semantic_recoding": round(float(X[col].isna().mean() * 100), 4),
            "unique_nonmissing_values": int(X[col].nunique(dropna=True)),
        })
    return pd.DataFrame(rows).sort_values(
        "missing_count_after_semantic_recoding", ascending=False
    )


def split_by_patient(X, y, groups):
    """Return row indices for disjoint patient-level train/validation/test sets."""
    outer = GroupShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    trainval_idx, test_idx = next(outer.split(X, y, groups=groups))

    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_FRACTION_OF_TRAINVAL,
        random_state=RANDOM_STATE + 1,
    )
    train_rel, val_rel = next(inner.split(
        X.iloc[trainval_idx],
        y.iloc[trainval_idx],
        groups=groups[trainval_idx],
    ))

    return trainval_idx[train_rel], trainval_idx[val_rel], test_idx


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_path = find_pruned_input()

    # Preserve the patient ID as text, and disable pandas' implicit NA parsing
    # so blank laboratory cells can be handled intentionally below.
    df = pd.read_csv(
        input_path,
        dtype={GROUP_COL: "string"},
        keep_default_na=False,
        low_memory=False,
    )

    make_input_audit(df).to_csv(
        OUTPUT_DIR / "preprocessing_input_audit.csv", index=False
    )

    required = [TARGET_COL, GROUP_COL, *NUMERIC_FEATURES, *CATEGORICAL_CODE_FEATURES]
    missing_required = [col for col in required if col not in df.columns]
    if missing_required:
        raise ValueError(f"Required columns missing from input: {missing_required}")

    # Keep the documented meaning of these blank test results.
    for col in NOT_MEASURED_FEATURES:
        if col in df.columns:
            text = df[col].astype("string").str.strip()
            mask = (
                text.isna()
                | text.eq("")
                | text.str.casefold().eq("none")
            )
            df.loc[mask.fillna(False), col] = "Not measured"

    # For other text columns, blank strings and '?' are treated as missing;
    # labels such as Unknown and Missing remain valid categories.
    for col in df.columns:
        if col in NOT_MEASURED_FEATURES:
            continue
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = clean_text_series(df[col])

    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    if y.isna().any() or not set(y.unique()).issubset({0, 1}):
        raise ValueError(f"{TARGET_COL} must be complete and contain only 0/1.")
    y = y.astype("int8")

    groups_series = df[GROUP_COL].astype("string").str.strip()
    if groups_series.isna().any() or groups_series.eq("").any():
        raise ValueError("patient_nbr contains missing or blank values.")
    groups = groups_series.astype(str).to_numpy()

    # Apply exclusions defensively even if the wrong earlier version of the
    # pruned file was copied into the dataset folder.
    drug_columns_removed_here = [
        col for col in DRUG_FEATURES_TO_EXCLUDE if col in df.columns
    ]
    identifier_columns = [col for col in [GROUP_COL, "encounter_id"] if col in df.columns]
    excluded_columns = list(dict.fromkeys([TARGET_COL, *identifier_columns, *drug_columns_removed_here]))
    X = df.drop(columns=excluded_columns).copy()

    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Category-code IDs are deliberately treated as categorical text.
    for col in CATEGORICAL_CODE_FEATURES:
        X[col] = clean_text_series(X[col])

    numeric_cols = list(NUMERIC_FEATURES)
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    for col in categorical_cols:
        X[col] = clean_text_series(X[col])

    make_feature_audit(X, numeric_cols).to_csv(
        OUTPUT_DIR / "preprocessing_feature_audit.csv", index=False
    )

    # Split first; all learned preprocessing is fitted on training only.
    train_idx, val_idx, test_idx = split_by_patient(X, y, groups)

    group_sets = [set(groups[idx]) for idx in (train_idx, val_idx, test_idx)]
    if not (
        group_sets[0].isdisjoint(group_sets[1])
        and group_sets[0].isdisjoint(group_sets[2])
        and group_sets[1].isdisjoint(group_sets[2])
    ):
        raise RuntimeError("Patient leakage detected across splits.")

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

    X_train = preprocessor.fit_transform(X.iloc[train_idx])
    X_validation = preprocessor.transform(X.iloc[val_idx])
    X_test = preprocessor.transform(X.iloc[test_idx])

    def to_csr(matrix):
        return matrix.tocsr() if sparse.issparse(matrix) else sparse.csr_matrix(matrix)

    X_train, X_validation, X_test = map(to_csr, (X_train, X_validation, X_test))
    sparse.save_npz(OUTPUT_DIR / "X_train.npz", X_train)
    sparse.save_npz(OUTPUT_DIR / "X_validation.npz", X_validation)
    sparse.save_npz(OUTPUT_DIR / "X_test.npz", X_test)

    for name, idx in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
        pd.DataFrame({
            "source_row_index": idx,
            TARGET_COL: y.iloc[idx].to_numpy(),
        }).to_csv(OUTPUT_DIR / f"y_{name}.csv", index=False)

    pd.DataFrame({
        "encoded_feature_name": preprocessor.get_feature_names_out()
    }).to_csv(OUTPUT_DIR / "encoded_feature_names.csv", index=False)
    joblib.dump(preprocessor, OUTPUT_DIR / "preprocessor.joblib")

    split_name_by_row = np.full(len(df), "", dtype=object)
    split_name_by_row[train_idx] = "train"
    split_name_by_row[val_idx] = "validation"
    split_name_by_row[test_idx] = "test"
    pd.DataFrame({
        "source_row_index": np.arange(len(df)),
        GROUP_COL: groups,
        TARGET_COL: y.to_numpy(),
        "split": split_name_by_row,
    }).to_csv(OUTPUT_DIR / "split_assignments.csv", index=False)

    split_rows = []
    for name, idx in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
        yi = y.iloc[idx]
        split_rows.append({
            "split": name,
            "encounters": len(idx),
            "unique_patients": pd.Series(groups[idx]).nunique(),
            "early_readmissions": int(yi.sum()),
            "early_readmission_rate_pct": round(float(yi.mean() * 100), 4),
        })
    split_summary = pd.DataFrame(split_rows)
    split_summary.to_csv(OUTPUT_DIR / "split_summary.csv", index=False)

    metadata = {
        "input_file": input_path.name,
        "input_shape": [int(df.shape[0]), int(df.shape[1])],
        "target": TARGET_COL,
        "patient_group_column": GROUP_COL,
        "numeric_features": numeric_cols,
        "categorical_code_features": CATEGORICAL_CODE_FEATURES,
        "categorical_features": categorical_cols,
        "drug_features_excluded_if_present": DRUG_FEATURES_TO_EXCLUDE,
        "drug_features_removed_by_this_run": drug_columns_removed_here,
        "not_measured_features": NOT_MEASURED_FEATURES,
        "split_method": "GroupShuffleSplit by patient_nbr",
        "random_state": RANDOM_STATE,
        "encoded_feature_count": int(X_train.shape[1]),
        "source_rows_preserved_across_splits": int(len(df)),
        "model_training_performed": False,
    }
    with open(OUTPUT_DIR / "preprocessing_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n========== PREPROCESSING COMPLETE ==========")
    print(f"Input: {input_path.name}")
    print(f"Rows preserved: {len(df):,}")
    print(f"Input columns: {df.shape[1]}")
    print(f"Predictor columns before encoding: {X.shape[1]}")
    print(f"Numerical predictors: {len(numeric_cols)}")
    print(f"Categorical predictors: {len(categorical_cols)}")
    print(f"Encoded features: {X_train.shape[1]}")
    print("Deleted drug columns excluded if present:", drug_columns_removed_here or "None; already absent")
    print("\nSplit summary:")
    print(split_summary.to_string(index=False))
    print(f"\nOutput folder: {OUTPUT_DIR}")
    print("Original input CSV was not modified. No model was trained.")


if __name__ == "__main__":
    main()
