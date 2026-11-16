
from pathlib import Path
import pandas as pd

# ============================================================
# CARE CONTINUUM: DRUG FEATURE VARIATION ANALYSIS
# ============================================================

DATASET_DIR = Path("dataset")

# Use the most processed file that currently exists.
INPUT_CANDIDATES = [
    DATASET_DIR / "diabetic_data_specialty_grouped.csv",
    DATASET_DIR / "diabetic_data_diagnosis_grouped.csv",
    DATASET_DIR / "diabetic_data_cleaned.csv",
]

REPORT_PATH = DATASET_DIR / "drug_feature_variation_report.csv"
OUTPUT_PATH = DATASET_DIR / "diabetic_data_drugs_pruned.csv"

# Remove drug features whose non-dominant categories account
# for less than 0.5% of non-missing encounters.
MIN_VARIATION_PCT = 0.5

# Medication columns in the UCI Diabetes dataset.
DRUG_FEATURES = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

EXPECTED_STATUSES = ["No", "Steady", "Up", "Down"]


def find_input_file():
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path

    expected = "\n".join(str(p) for p in INPUT_CANDIDATES)
    raise FileNotFoundError(
        "No input dataset was found. Checked:\n" + expected
    )


def analyze_feature(df, column):
    # Normalize whitespace for analysis without changing the dataset.
    values = df[column].astype("string").str.strip()
    observed = values.dropna()

    row_count = len(values)
    observed_count = len(observed)
    missing_count = int(values.isna().sum())

    missing_pct = (
        missing_count / row_count * 100
        if row_count else 0.0
    )

    category_counts = observed.value_counts()
    unique_count = int(observed.nunique())

    if observed_count == 0:
        dominant_value = ""
        dominant_pct = 0.0
        variation_pct = 0.0
        decision = "REMOVE"
        reason = "All values are missing"

    else:
        dominant_value = str(category_counts.index[0])
        dominant_count = int(category_counts.iloc[0])

        dominant_pct = dominant_count / observed_count * 100

        # Percentage of observed encounters NOT in the
        # most common category.
        variation_pct = (
            (observed_count - dominant_count)
            / observed_count * 100
        )

        if unique_count <= 1:
            decision = "REMOVE"
            reason = "Constant feature"

        elif variation_pct < MIN_VARIATION_PCT:
            decision = "REMOVE"
            reason = (
                f"Near-constant: variation below "
                f"{MIN_VARIATION_PCT}%"
            )

        else:
            decision = "KEEP"
            reason = "Variation meets threshold"

    # Report category counts for inspection.
    distribution = "; ".join(
        f"{status}={int(category_counts.get(status, 0))}"
        for status in EXPECTED_STATUSES
    )

    # Report explicit category percentages.
    status_pct = {}
    for status in EXPECTED_STATUSES:
        count = int(category_counts.get(status, 0))
        status_pct[f"{status.lower()}_pct"] = (
            count / observed_count * 100
            if observed_count else 0.0
        )

    return {
        "feature": column,
        "total_encounters": row_count,
        "observed_encounters": observed_count,
        "missing_count": missing_count,
        "missing_pct": round(missing_pct, 4),
        "unique_values": unique_count,
        "dominant_value": dominant_value,
        "dominant_value_pct": round(dominant_pct, 4),
        "variation_from_dominant_pct": round(variation_pct, 4),
        "category_counts_No_Steady_Up_Down": distribution,
        **{
            key: round(value, 4)
            for key, value in status_pct.items()
        },
        "decision": decision,
        "reason": reason,
    }


def main():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    input_path = find_input_file()
    print(f"Reading dataset: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)

    # Keep patient identifiers as strings when present.
    if "patient_nbr" in df.columns:
        df["patient_nbr"] = df["patient_nbr"].astype("string")

    drug_columns = [
        column for column in DRUG_FEATURES
        if column in df.columns
    ]

    missing_columns = [
        column for column in DRUG_FEATURES
        if column not in df.columns
    ]

    if not drug_columns:
        raise ValueError(
            "No expected drug columns were found. "
            "Check the input dataset and column names."
        )

    print(f"\nDataset shape before reduction: {df.shape}")
    print(f"Drug features found: {len(drug_columns)}")

    if missing_columns:
        print(
            "\nDrug columns not found in the input "
            "(possibly already removed):"
        )
        print(", ".join(missing_columns))

    # Analyze only the medication columns.
    results = [
        analyze_feature(df, column)
        for column in drug_columns
    ]

    report = pd.DataFrame(results)

    report = report.sort_values(
        by="variation_from_dominant_pct",
        ascending=True,
    )

    report.to_csv(REPORT_PATH, index=False)

    # Remove only drug columns flagged by our stated rule.
    remove_columns = report.loc[
        report["decision"] == "REMOVE", "feature"
    ].tolist()

    reduced_df = df.drop(columns=remove_columns)
    reduced_df.to_csv(OUTPUT_PATH, index=False)

    # Print a readable summary.
    display_columns = [
        "feature",
        "unique_values",
        "dominant_value",
        "dominant_value_pct",
        "variation_from_dominant_pct",
        "decision",
        "reason",
    ]

    print("\n========== DRUG FEATURE REPORT ==========")
    print(
        report[display_columns].to_string(
            index=False
        )
    )

    print("\n========== SUMMARY ==========")
    print(f"Original rows: {len(df):,}")
    print(f"Original columns: {df.shape[1]}")
    print(f"Drug columns analyzed: {len(drug_columns)}")
    print(f"Drug columns removed: {len(remove_columns)}")
    print(
        f"Drug columns retained: "
        f"{len(drug_columns) - len(remove_columns)}"
    )
    print(f"Final columns: {reduced_df.shape[1]}")

    print("\nRemoved medication features:")
    if remove_columns:
        for column in remove_columns:
            print(f"  - {column}")
    else:
        print("  None")

    print(f"\nVariation report saved to: {REPORT_PATH}")
    print(f"Reduced dataset saved to: {OUTPUT_PATH}")
    print("\nThe input dataset was not overwritten.")


if __name__ == "__main__":
    main()
