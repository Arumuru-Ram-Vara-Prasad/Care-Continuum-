"""Level 1 rule-based cleaning for Care Continuum.

Run from the Care Continuum project folder:
    python clean_data.py

Default input:   dataset/diabetic_data.xlsx (falls back to dataset/diabetic_data.csv)
Default output:  dataset/diabetic_data_cleaned.csv
Default audit:   data_cleaning_audit.txt

This script performs only the foundational, deterministic cleaning decisions.
Diagnosis grouping, medication-signal analysis, train/test splitting, imputation
outside the three approved Unknown fields, encoding, scaling, and model training
are intentionally left for later steps.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import re
import sys
from typing import Optional

import pandas as pd

# Decisions approved for this foundational cleaning pass.
DEATH_DISPOSITION_CODES = {"11", "19", "20", "21"}
VALID_READMISSION_LABELS = {"NO", ">30", "<30"}
DROP_COLUMNS = ["encounter_id", "weight"]
EXPECTED_CONSTANT_COLUMNS = ["examide", "citoglipton"]
UNKNOWN_FOR_MISSING_COLUMNS = ["race", "payer_code", "medical_specialty"]
CATEGORICAL_ID_COLUMNS = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]
DIAGNOSIS_COLUMNS = ["diag_1", "diag_2", "diag_3"]


class CleaningError(RuntimeError):
    """Raised when the source data fails an important validation check."""


def find_default_input(project_dir: Path) -> Path:
    for relative in ("dataset/diabetic_data.xlsx", "dataset/diabetic_data.csv"):
        candidate = project_dir / relative
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find dataset/diabetic_data.xlsx or dataset/diabetic_data.csv. "
        "Run this script from the project folder or pass --input."
    )


def find_mapping_file(project_dir: Path) -> Optional[Path]:
    for candidate in (
        project_dir / "IDS_mapping.csv",
        project_dir / "dataset" / "IDS_mapping.csv",
    ):
        if candidate.exists():
            return candidate
    return None


def normalize_code(value):
    """Normalize numeric codes represented as 11, '11', or '11.0'."""
    if pd.isna(value):
        return pd.NA
    value = str(value).strip()
    if value in {"", "?"}:
        return pd.NA
    if re.fullmatch(r"\d+\.0+", value):
        value = value.split(".", 1)[0]
    return value


def load_dataset(path: Path) -> pd.DataFrame:
    """Load data while preserving the literal category string 'None'."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    print(f"Loading dataset: {path}", flush=True)
    # IDs are identifiers, not measurements. Read patient_nbr as text so
    # Python preserves its exact identifier representation throughout cleaning.
    dtype = {col: "string" for col in [*DIAGNOSIS_COLUMNS, "patient_nbr"]}
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        df = pd.read_excel(path, dtype=dtype, na_values=["?"], keep_default_na=False)
    elif suffix == ".csv":
        df = pd.read_csv(
            path, dtype=dtype, na_values=["?"], keep_default_na=False, low_memory=False
        )
    else:
        raise ValueError("Input must be a CSV or Excel file (.csv, .xlsx, .xlsm, .xls).")

    # Normalize whitespace and missing markers consistently without treating
    # the literal text 'None' as missing.
    for col in df.select_dtypes(include=["object", "string"]).columns:
        values = df[col].astype("string").str.strip()
        df[col] = values.replace({"": pd.NA, "?": pd.NA})
    return df


def parse_discharge_mapping(path: Optional[Path]) -> dict[str, str]:
    """Parse the multi-section IDS_mapping.csv and return discharge descriptions."""
    if path is None:
        raise CleaningError(
            "IDS_mapping.csv was not found. Place it in the project root or dataset folder; "
            "the script needs it to validate death-related exclusions."
        )

    table = pd.read_csv(path, keep_default_na=False, dtype=str)
    if table.shape[1] < 2:
        raise CleaningError(f"Could not parse the mapping file: {path}")

    descriptions: dict[str, str] = {}
    in_discharge_section = False
    for _, row in table.iterrows():
        first = str(row.iloc[0]).strip()
        second = str(row.iloc[1]).strip()

        if first == "discharge_disposition_id" and second.lower() == "description":
            in_discharge_section = True
            continue
        if first.endswith("_id") and second.lower() == "description":
            in_discharge_section = first == "discharge_disposition_id"
            continue
        if in_discharge_section and first.isdigit() and second:
            descriptions[normalize_code(first)] = second

    print(f"Read discharge descriptions from: {path}", flush=True)

    # Fail closed: do not exclude records based on hard-coded codes unless the
    # mapping file confirms each code is described as expired.
    for code in sorted(DEATH_DISPOSITION_CODES, key=int):
        description = descriptions.get(code)
        if not description:
            raise CleaningError(
                f"Discharge code {code} is missing from IDS_mapping.csv; "
                "cannot safely validate the exclusion policy."
            )
        if "expired" not in description.casefold():
            raise CleaningError(
                f"Discharge code {code} is not described as expired in IDS_mapping.csv: "
                f"{description!r}. Review the exclusion list before proceeding."
            )
        print(f"  Verified code {code}: {description}", flush=True)

    return descriptions


def format_label_counts(labels: pd.Series) -> list[str]:
    counts = labels.value_counts(dropna=False)
    total = len(labels)
    lines = ["Readmission labels:"]
    for label in ["NO", ">30", "<30"]:
        count = int(counts.get(label, 0))
        pct = 100 * count / total if total else 0
        lines.append(f"  {label}: {count:,} ({pct:.2f}%)")
    return lines


def clean_data(
    input_path: Path,
    output_path: Path,
    audit_path: Path,
    mapping_path: Optional[Path],
) -> None:
    if input_path.resolve() == output_path.resolve():
        raise CleaningError("Output path must differ from the original input path.")

    df = load_dataset(input_path)
    original_rows, original_cols = df.shape
    print(f"Loaded {original_rows:,} rows and {original_cols:,} columns.", flush=True)

    required = {
        "patient_nbr", "readmitted", "discharge_disposition_id", *DIAGNOSIS_COLUMNS
    }
    missing_required = required - set(df.columns)
    if missing_required:
        raise CleaningError(f"Required columns are missing: {sorted(missing_required)}")
    if df["patient_nbr"].isna().any():
        raise CleaningError(
            "patient_nbr contains missing values. Resolve these before patient-grouped splitting."
        )

    original_labels = df["readmitted"].astype("string").str.strip()
    invalid_labels = original_labels.isna() | ~original_labels.isin(VALID_READMISSION_LABELS)
    if invalid_labels.any():
        examples = original_labels[invalid_labels].value_counts(dropna=False).head(10).to_dict()
        raise CleaningError(
            "The readmitted column contains missing or unexpected labels. "
            f"Inspect these values before cleaning: {examples}"
        )

    # Validate the exclusion codes against the supplied mapping file first.
    mapping = parse_discharge_mapping(mapping_path)
    disposition = df["discharge_disposition_id"].map(normalize_code).astype("string")
    excluded_mask = disposition.isin(DEATH_DISPOSITION_CODES)
    excluded_rows = int(excluded_mask.sum())

    audit: list[str] = [
        "CARE CONTINUUM — LEVEL 1 DATA CLEANING AUDIT",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Input file: {input_path.resolve()}",
        f"Output file: {output_path.resolve()}",
        f"Mapping file: {mapping_path.resolve() if mapping_path else 'not found'}",
        "",
        f"Original shape: {original_rows:,} rows × {original_cols:,} columns",
    ]
    audit.extend(format_label_counts(original_labels))
    audit.append("")
    audit.append("Death-related exclusion codes verified against IDS_mapping.csv:")
    for code in sorted(DEATH_DISPOSITION_CODES, key=int):
        n_code = int(disposition.eq(code).sum())
        labels_for_code = original_labels[disposition.eq(code)].value_counts().to_dict()
        audit.append(
            f"  Code {code}: {mapping[code]}; rows={n_code:,}; readmission labels={labels_for_code}"
        )
    audit.append(f"Total death-related encounters excluded: {excluded_rows:,}")
    audit.append("Hospice codes 13 and 14 are retained.")
    audit.append("")

    # Apply the agreed row exclusion and create the binary target.
    df = df.loc[~excluded_mask].copy()
    retained_labels = original_labels.loc[~excluded_mask]
    df["readmitted_30d"] = retained_labels.eq("<30").astype("int8")
    df.drop(columns=["readmitted"], inplace=True)

    # Drop only the columns already agreed for the basic cleaning pass.
    dropped_columns: list[str] = []
    for col in DROP_COLUMNS:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            dropped_columns.append(col)

    # Drop the two expected constants only when the actual data confirms it.
    for col in EXPECTED_CONSTANT_COLUMNS:
        if col not in df.columns:
            audit.append(f"NOTE: Expected constant column {col!r} was not present.")
            continue
        if df[col].nunique(dropna=False) <= 1:
            df.drop(columns=[col], inplace=True)
            dropped_columns.append(col)
            audit.append(f"Confirmed and dropped constant column: {col}")
        else:
            audit.append(f"WARNING: {col} was not constant; retained for review.")

    # Only the three approved columns receive Unknown for missing values.
    unknown_counts: dict[str, int] = {}
    for col in UNKNOWN_FOR_MISSING_COLUMNS:
        if col in df.columns:
            unknown_counts[col] = int(df[col].isna().sum())
            df[col] = df[col].astype("string").fillna("Unknown")

    # The invalid gender category is reported, not silently recoded. Handling it
    # remains an explicit decision for review. Truly missing gender also remains NA.
    if "gender" in df.columns:
        gender_values = df["gender"].astype("string").str.strip()
        invalid_gender_count = int(
            gender_values.str.casefold().eq("unknown/invalid").fillna(False).sum()
        )
        missing_gender_count = int(gender_values.isna().sum())
        audit.append(f"Gender 'Unknown/Invalid' records preserved unchanged: {invalid_gender_count:,}")
        audit.append(f"Truly missing gender values left missing for later review: {missing_gender_count:,}")

    # Represent ID codes as categories, not continuous measurements; no encoding yet.
    for col in CATEGORICAL_ID_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(normalize_code).astype("string")

    # Diagnosis grouping was approved as a planned transformation, but this
    # Level 1 script preserves raw diagnoses only. Grouping will be implemented
    # and checked as a separate feature-engineering step.

    # patient_nbr is intentionally retained for future grouped splitting only.
    if "patient_nbr" not in df.columns:
        raise CleaningError("patient_nbr was unexpectedly removed; grouped splitting needs it.")

    target = df.pop("readmitted_30d")
    df["readmitted_30d"] = target

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(output_path, index=False, encoding="utf-8")
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot write {output_path}. Close it if open in Excel or choose another --output path."
        ) from exc

    final_rows, final_cols = df.shape
    unique_patients = int(df["patient_nbr"].nunique(dropna=True))
    patient_visits = df["patient_nbr"].value_counts(dropna=True)
    patients_with_repeats = int((patient_visits > 1).sum())
    max_visits = int(patient_visits.max()) if not patient_visits.empty else 0
    target_counts = df["readmitted_30d"].value_counts().to_dict()

    audit.extend([
        "",
        "LEVEL 1 RULES APPLIED",
        "Dropped columns: " + (", ".join(dropped_columns) if dropped_columns else "none"),
        "Missing race/payer_code/medical_specialty values replaced with 'Unknown':",
    ])
    for col, count in unknown_counts.items():
        audit.append(f"  {col}: {count:,}")
    audit.append("patient_nbr retained as a string identifier for grouping; exclude it from predictive features.")
    audit.append("ID-code columns converted to string categories; not one-hot encoded.")
    audit.append("Original diagnosis code columns preserved unchanged; diagnosis grouping deferred.")
    audit.append("Medication columns retained except examide/citoglipton if confirmed constant.")
    audit.append("No medication signal scoring/thresholding was performed.")
    audit.append("No deduplication, train/test split, imputation beyond approved Unknown fields, encoding, scaling, or model training was performed.")
    audit.append("Original readmitted target was removed; use readmitted_30d as y.")
    audit.append("")
    audit.append(f"Cleaned shape: {final_rows:,} rows × {final_cols:,} columns")
    audit.append(f"Rows excluded: {original_rows - final_rows:,}")
    audit.append(f"Unique patients remaining: {unique_patients:,}")
    audit.append(f"Patients with 2+ remaining encounters: {patients_with_repeats:,}")
    audit.append(f"Maximum remaining encounters for one patient: {max_visits:,}")
    audit.append("")
    audit.append("Binary target counts (readmitted_30d):")
    audit.append(f"  0 (original NO or >30): {int(target_counts.get(0, 0)):,}")
    audit.append(f"  1 (original <30): {int(target_counts.get(1, 0)):,}")
    audit.append("")
    audit.append("IMPORTANT for later stages:")
    audit.append("  * Use readmitted_30d as y and never include it in X.")
    audit.append("  * Keep patient_nbr only for grouped splitting; exclude it from X.")
    audit.append("  * Raw diag_1/diag_2/diag_3 are retained; diagnosis grouping is a later step.")
    audit.append("  * Medication features require the separately planned signal/usefulness experiment.")

    audit_path.write_text("\n".join(audit) + "\n", encoding="utf-8")

    print("\nLevel 1 cleaning complete.", flush=True)
    print(f"Rows before cleaning: {original_rows:,}")
    print(f"Death-related rows removed: {excluded_rows:,}")
    print(f"Rows in cleaned dataset: {final_rows:,}")
    print(f"Columns in cleaned dataset: {final_cols:,}")
    print(f"Unique patients remaining: {unique_patients:,}")
    print(f"Patients with 2+ remaining encounters: {patients_with_repeats:,}")
    print(f"Saved cleaned dataset: {output_path.resolve()}")
    print(f"Saved audit report: {audit_path.resolve()}")
    print("\nBinary target distribution:")
    print(df["readmitted_30d"].value_counts().sort_index().rename(index={
        0: "Not readmitted within 30 days", 1: "Readmitted within 30 days"
    }).to_string())
    print("\nGender 'Unknown/Invalid' was reported, not recoded.")
    print("Diagnosis grouping and medication-signal analysis are deferred to their own steps.")


def main() -> int:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Apply foundational Level 1 Care Continuum data-cleaning rules."
    )
    parser.add_argument("--input", type=Path, default=None, help="Input CSV/Excel file path.")
    parser.add_argument("--output", type=Path, default=project_dir / "dataset" / "diabetic_data_cleaned.csv")
    parser.add_argument("--audit", type=Path, default=project_dir / "data_cleaning_audit.txt")
    parser.add_argument("--mapping", type=Path, default=None, help="Path to IDS_mapping.csv.")
    args = parser.parse_args()

    input_path = args.input if args.input is not None else find_default_input(project_dir)
    if not input_path.is_absolute():
        input_path = (Path.cwd() / input_path).resolve()
    output_path = args.output if args.output.is_absolute() else (Path.cwd() / args.output).resolve()
    audit_path = args.audit if args.audit.is_absolute() else (Path.cwd() / args.audit).resolve()

    mapping_path = args.mapping
    if mapping_path is not None and not mapping_path.is_absolute():
        mapping_path = (Path.cwd() / mapping_path).resolve()
    if mapping_path is None:
        mapping_path = find_mapping_file(project_dir)
    elif not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    try:
        clean_data(input_path, output_path, audit_path, mapping_path)
    except (CleaningError, FileNotFoundError, ValueError, PermissionError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
