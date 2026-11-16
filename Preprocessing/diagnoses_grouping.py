
"""
Care Continuum - Group diagnosis codes

Input:
    dataset/diabetic_data_cleaned.csv

Outputs:
    dataset/diabetic_data_diagnosis_grouped.csv
    diagnosis_grouping_summary.csv

Actions:
    1. Group diag_1, diag_2, diag_3 into nine broad groups.
    2. Keep missing diagnoses in a separate "Missing" group.
    3. Remove the original diagnosis columns from the OUTPUT.
    4. Preserve the original input file.
"""

from pathlib import Path
import re
import sys

import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    PROJECT_DIR / "dataset" / "diabetic_data_cleaned.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR / "dataset" / "diabetic_data_diagnosis_grouped.csv"
)

SUMMARY_FILE = (
    PROJECT_DIR / "diagnosis_grouping_summary.csv"
)

DIAGNOSIS_COLUMNS = ["diag_1", "diag_2", "diag_3"]

GROUP_NAMES = [
    "Circulatory",
    "Respiratory",
    "Digestive",
    "Diabetes",
    "Injury",
    "Musculoskeletal",
    "Genitourinary",
    "Neoplasms",
    "Other",
    "Missing",
]


# --------------------------------------------------
# Step 1: Normalize diagnosis codes
# --------------------------------------------------

def normalize_diagnosis(value):
    """Convert a diagnosis value into a clean string."""

    if pd.isna(value):
        return None

    code = str(value).strip().upper()

    if code in {"", "?", "NAN", "<NA>"}:
        return None

    # Convert values such as 250.0 to 250,
    # without changing meaningful codes such as 250.83.
    if re.fullmatch(r"\d+\.0+", code):
        code = code.split(".", 1)[0]

    return code


# --------------------------------------------------
# Step 2: Map each diagnosis to a broad group
# --------------------------------------------------

def group_diagnosis(value):
    """Map an ICD-9 diagnosis into one of nine groups."""

    code = normalize_diagnosis(value)

    if code is None:
        return "Missing"

    # V-codes and E-codes are assigned to Other
    # under our agreed grouping scheme.
    if code.startswith(("V", "E")):
        return "Other"

    # Accept numeric ICD-9 codes, with optional decimals.
    match = re.fullmatch(r"(\d{1,3})(?:\.\d+)?", code)

    if not match:
        return "Other"

    category = int(match.group(1))

    # The order and ranges follow our proposed grouping.
    if category == 250:
        return "Diabetes"

    if 390 <= category <= 459 or category == 785:
        return "Circulatory"

    if 460 <= category <= 519 or category == 786:
        return "Respiratory"

    if 520 <= category <= 579 or category == 787:
        return "Digestive"

    if 580 <= category <= 629 or category == 788:
        return "Genitourinary"

    if 140 <= category <= 239:
        return "Neoplasms"

    if 710 <= category <= 739:
        return "Musculoskeletal"

    if 800 <= category <= 999:
        return "Injury"

    return "Other"


# --------------------------------------------------
# Step 3: Load the cleaned dataset
# --------------------------------------------------

def main():

    if not INPUT_FILE.exists():
        print(f"ERROR: Dataset not found:\n{INPUT_FILE}")
        print(
            "Check that this script is in your project folder "
            "and that the dataset is inside the dataset folder."
        )
        return 1

    if INPUT_FILE.resolve() == OUTPUT_FILE.resolve():
        print("ERROR: Input and output files must be different.")
        return 1

    print("Loading cleaned dataset...", flush=True)

    df = pd.read_csv(
        INPUT_FILE,
        keep_default_na=False,
        low_memory=False,
        dtype={
            "patient_nbr": "string",
            "diag_1": "string",
            "diag_2": "string",
            "diag_3": "string",
        },
    )

    original_rows, original_cols = df.shape

    missing_columns = [
        col for col in DIAGNOSIS_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        print(
            f"ERROR: Missing required columns: {missing_columns}"
        )
        return 1

    print(
        f"Loaded {original_rows:,} encounters "
        f"and {original_cols} columns.",
        flush=True,
    )


    # --------------------------------------------------
    # Step 4: Create the three grouped columns
    # --------------------------------------------------

    summary_rows = []

    for col in DIAGNOSIS_COLUMNS:

        print(f"Grouping {col}...", flush=True)

        original_codes = df[col].map(normalize_diagnosis)

        group_col = f"{col}_group"

        df[group_col] = (
            df[col]
            .map(group_diagnosis)
            .astype("string")
        )

        # Record counts for the summary file.
        for group in GROUP_NAMES:

            mask = df[group_col].eq(group)

            summary_rows.append({
                "diagnosis_column": col,
                "group": group,
                "encounters": int(mask.sum()),
                "distinct_original_codes": int(
                    original_codes[mask].nunique()
                ),
            })

        # Every encounter must receive exactly one group.
        if df[group_col].isna().any():
            print(
                f"ERROR: Missing group assignments in {group_col}."
            )
            return 1


    # --------------------------------------------------
    # Step 5: Remove original diagnosis columns
    # --------------------------------------------------

    df.drop(columns=DIAGNOSIS_COLUMNS, inplace=True)


    # --------------------------------------------------
    # Step 6: Validate and save the outputs
    # --------------------------------------------------

    grouped_columns = [
        "diag_1_group",
        "diag_2_group",
        "diag_3_group",
    ]

    if any(col in df.columns for col in DIAGNOSIS_COLUMNS):
        print("ERROR: Original diagnosis columns still exist.")
        return 1

    if not all(col in df.columns for col in grouped_columns):
        print("ERROR: A grouped diagnosis column is missing.")
        return 1

    if len(df) != original_rows:
        print("ERROR: The number of encounters changed.")
        return 1

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8",
    )


    # --------------------------------------------------
    # Step 7: Print a final report
    # --------------------------------------------------

    print("\n" + "=" * 55)
    print("DIAGNOSIS GROUPING COMPLETED")
    print("=" * 55)

    print(f"Original rows: {original_rows:,}")
    print(f"Output rows:   {len(df):,}")
    print(f"Original columns: {original_cols}")
    print(f"Output columns:   {len(df.columns)}")

    print("\nNew diagnosis columns:")
    for col in grouped_columns:
        print(f"  - {col}")

    print("\nOriginal diagnosis columns removed:")
    for col in DIAGNOSIS_COLUMNS:
        print(f"  - {col}")

    print("\nGroup counts:")

    for col in grouped_columns:
        print(f"\n{col}:")

        print(
            df[col]
            .value_counts()
            .reindex(GROUP_NAMES, fill_value=0)
            .to_string()
        )

    print(f"\nGrouped dataset saved to:\n{OUTPUT_FILE}")
    print(f"\nSummary saved to:\n{SUMMARY_FILE}")

    print("\nThe original cleaned dataset was not modified.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
