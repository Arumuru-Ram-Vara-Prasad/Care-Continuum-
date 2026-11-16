
"""
Care Continuum: Group medical specialties into broader categories.

Input:
    dataset/diabetic_data_diagnosis_grouped.csv

Outputs:
    dataset/diabetic_data_specialty_grouped.csv
    medical_specialty_group_summary.csv

The input file is never modified.
The original medical_specialty column is replaced in the output.
"""

from pathlib import Path
import sys
import pandas as pd


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR / "dataset" / "diabetic_data_diagnosis_grouped.csv"
)

OUTPUT_FILE = (
    BASE_DIR / "dataset" / "diabetic_data_specialty_grouped.csv"
)

SUMMARY_FILE = (
    BASE_DIR / "medical_specialty_group_summary.csv"
)


# --------------------------------------------------
# 2. Define the specialty groups
# --------------------------------------------------

EXACT_GROUPS = {
    "Unknown": "Unknown",
    "PhysicianNotFound": "Unknown",

    # General medicine
    "InternalMedicine": "General/Internal Medicine",
    "Family/GeneralPractice": "General/Internal Medicine",
    "Hospitalist": "General/Internal Medicine",
    "Osteopath": "General/Internal Medicine",

    # Emergency care
    "Emergency/Trauma": "Emergency Medicine",

    # Cardiovascular
    "Cardiology": "Cardiology/Vascular",
    "Surgery-Cardiovascular/Thoracic": "Cardiology/Vascular",
    "Surgery-Cardiovascular": "Cardiology/Vascular",
    "Surgery-Vascular": "Cardiology/Vascular",

    # Surgery
    "Surgery-General": "Surgery",
    "Surgery-Neuro": "Surgery",
    "Surgery-Thoracic": "Surgery",
    "Surgeon": "Surgery",
    "SurgicalSpecialty": "Surgery",
    "Surgery-Plastic": "Surgery",
    "Surgery-Colon&Rectal": "Surgery",
    "Surgery-Maxillofacial": "Surgery",
    "Surgery-PlasticwithinHeadandNeck": "Surgery",

    # Musculoskeletal and rehabilitation
    "Orthopedics": "Orthopedics/Rehabilitation",
    "Orthopedics-Reconstructive": "Orthopedics/Rehabilitation",
    "PhysicalMedicineandRehabilitation": "Orthopedics/Rehabilitation",
    "Podiatry": "Orthopedics/Rehabilitation",
    "SportsMedicine": "Orthopedics/Rehabilitation",

    # Respiratory
    "Pulmonology": "Respiratory",

    # Kidney and urinary system
    "Nephrology": "Renal/Urinary",
    "Urology": "Renal/Urinary",

    # Digestive system
    "Gastroenterology": "Digestive",
    "Proctology": "Digestive",

    # Neurology
    "Neurology": "Neurology",
    "Neurophysiology": "Neurology",

    # Mental health
    "Psychiatry": "Mental Health",
    "Psychology": "Mental Health",
    "Psychiatry-Child/Adolescent": "Mental Health",
    "Psychiatry-Addictive": "Mental Health",

    # Women's health
    "ObstetricsandGynecology": "Women's Health",
    "Gynecology": "Women's Health",
    "Obsterics&Gynecology-GynecologicOnco": "Women's Health",
    "Obstetrics": "Women's Health",
    "Perinatology": "Women's Health",

    # Cancer and blood disorders
    "Oncology": "Oncology/Hematology",
    "Hematology/Oncology": "Oncology/Hematology",
    "Hematology": "Oncology/Hematology",

    # Endocrine system
    "Endocrinology": "Endocrine/Metabolic",
    "Endocrinology-Metabolism": "Endocrine/Metabolic",

    # Diagnostic specialties and anesthesia
    "Radiologist": "Diagnostics/Anesthesia",
    "Radiology": "Diagnostics/Anesthesia",
    "Pathology": "Diagnostics/Anesthesia",
    "Anesthesiology": "Diagnostics/Anesthesia",
}


PEDIATRIC_VALUES = {
    "Pediatrics",
    "Pediatrics-Endocrinology",
    "Pediatrics-CriticalCare",
    "Pediatrics-Pulmonology",
    "Pediatrics-Neurology",
    "Pediatrics-Hematology-Oncology",
    "Pediatrics-AllergyandImmunology",
    "Pediatrics-EmergencyMedicine",
    "Pediatrics-InfectiousDiseases",
    "Anesthesiology-Pediatric",
    "Cardiology-Pediatric",
    "Surgery-Pediatric",
}


GROUP_ORDER = [
    "Unknown",
    "General/Internal Medicine",
    "Emergency Medicine",
    "Cardiology/Vascular",
    "Surgery",
    "Orthopedics/Rehabilitation",
    "Respiratory",
    "Renal/Urinary",
    "Digestive",
    "Neurology",
    "Mental Health",
    "Pediatrics",
    "Women's Health",
    "Oncology/Hematology",
    "Endocrine/Metabolic",
    "Diagnostics/Anesthesia",
    "Other Specialties",
]


# --------------------------------------------------
# 3. Convert each specialty to a group
# --------------------------------------------------

def group_specialty(value):
    if pd.isna(value):
        return "Unknown"

    value = str(value).strip()

    if value in {"", "?", "nan", "<NA>"}:
        return "Unknown"

    if value in PEDIATRIC_VALUES or value.startswith("Pediatrics-"):
        return "Pediatrics"

    if value in EXACT_GROUPS:
        return EXACT_GROUPS[value]

    # Group any other surgery-prefixed specialties together.
    if value.startswith("Surgery-"):
        return "Surgery"

    # Unmapped specialties are assigned to the residual group.
    return "Other Specialties"


# --------------------------------------------------
# 4. Load, group and validate the dataset
# --------------------------------------------------

def main():
    input_file = INPUT_FILE
    output_file = OUTPUT_FILE
    summary_file = SUMMARY_FILE

    # Optional file paths can be provided through PowerShell.
    if len(sys.argv) >= 2:
        input_file = Path(sys.argv[1]).expanduser().resolve()

    if len(sys.argv) >= 3:
        output_file = Path(sys.argv[2]).expanduser().resolve()

    if len(sys.argv) >= 4:
        summary_file = Path(sys.argv[3]).expanduser().resolve()

    if not input_file.exists():
        print(f"ERROR: Input file not found:\n{input_file}")
        print(
            "Expected: dataset/diabetic_data_diagnosis_grouped.csv"
        )
        return 1

    # Prevent accidental overwriting of the input file.
    if input_file.resolve() == output_file.resolve():
        print("ERROR: Input and output paths must be different.")
        return 1

    print(f"Loading dataset: {input_file}", flush=True)

    df = pd.read_csv(
        input_file,
        keep_default_na=False,
        low_memory=False,
        dtype={"patient_nbr": "string"},
    )

    if "medical_specialty" not in df.columns:
        print("ERROR: medical_specialty column not found.")
        return 1

    rows_before = len(df)

    original_specialties = (
        df["medical_specialty"]
        .replace({"": "Unknown", "?": "Unknown"})
        .nunique()
    )

    print(f"Encounters loaded: {rows_before:,}")
    print(f"Distinct original specialty labels: {original_specialties}")

    # Create the new grouped column.
    print("Grouping medical specialties...", flush=True)

    df["medical_specialty_group"] = (
        df["medical_specialty"].map(group_specialty)
    )

    # Remove the original high-cardinality column from the output.
    df.drop(columns=["medical_specialty"], inplace=True)

    # Validate the transformed data.
    if len(df) != rows_before:
        print("ERROR: The number of encounters changed.")
        return 1

    if df["medical_specialty_group"].isna().any():
        print("ERROR: Some specialties have no group assignment.")
        return 1

    actual_groups = df["medical_specialty_group"].nunique()

    if not df["medical_specialty_group"].isin(GROUP_ORDER).all():
        print("ERROR: Unexpected group label found.")
        return 1

    # Save the new dataset.
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8",
    )

    # Save the frequency summary.
    counts = (
        df["medical_specialty_group"]
        .value_counts()
        .reindex(GROUP_ORDER, fill_value=0)
    )

    summary = pd.DataFrame({
        "medical_specialty_group": counts.index,
        "encounters": counts.values,
        "percentage": (
            counts.values / len(df) * 100
        ).round(2),
    })

    summary_file.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        summary_file,
        index=False,
        encoding="utf-8",
    )

    # Print results.
    print("\n" + "=" * 55)
    print("MEDICAL SPECIALTY GROUPING COMPLETE")
    print("=" * 55)

    print(f"Encounters retained: {len(df):,}")
    print(f"Groups present: {actual_groups}")
    print("\nGroup frequencies:")
    print(summary.to_string(index=False))

    print(f"\nGrouped dataset saved to:\n{output_file}")
    print(f"Frequency summary saved to:\n{summary_file}")
    print("\nThe input dataset was not modified.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
