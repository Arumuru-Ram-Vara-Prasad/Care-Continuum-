
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


# ============================================================
# CARE CONTINUUM
# Verify readmission rates for DELETED drug features only.
# This script does not modify any dataset.
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"

PRUNED_DATASET = DATASET_DIR / "diabetic_data_drugs_pruned(3).csv"
PREVIOUS_REPORT = DATASET_DIR / "drug_feature_readmission_report.csv"

INPUT_CANDIDATES = [
    DATASET_DIR / "diabetic_data_specialty_grouped.csv",
    DATASET_DIR / "diabetic_data_diagnosis_grouped.csv",
    DATASET_DIR / "diabetic_data_cleaned.csv",
]

SUMMARY_OUTPUT = DATASET_DIR / "deleted_drug_readmission_verification.csv"
DETAIL_OUTPUT = DATASET_DIR / "deleted_drug_status_detail.csv"

MIN_GROUP_COUNT = 30
MIN_DIFFERENCE_PP = 2.0

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


def find_original_dataset():
    """Use the same input-file priority as the reduction script."""
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Original dataset not found. Expected one of:\n"
        + "\n".join(str(p) for p in INPUT_CANDIDATES)
    )


def get_target(df):
    """Return 1 for readmitted within 30 days, 0 otherwise."""
    if "readmitted_30d" in df.columns:
        raw = df["readmitted_30d"]
        numeric = pd.to_numeric(raw, errors="coerce")

        if numeric.notna().any() and set(
            numeric.dropna().unique()
        ).issubset({0, 1}):
            return numeric

        text = raw.astype("string").str.strip().str.lower()
        return text.map({
            "1": 1,
            "0": 0,
            "true": 1,
            "false": 0,
            "yes": 1,
            "no": 0,
        })

    if "readmitted" in df.columns:
        readmitted = (
            df["readmitted"]
            .astype("string")
            .str.strip()
            .str.upper()
        )

        return readmitted.map({
            "<30": 1,
            "NO": 0,
            ">30": 0,
        })

    raise ValueError(
        "The dataset must contain 'readmitted_30d' or 'readmitted'."
    )


def get_deleted_features(original_columns):
    """Identify drug columns absent from the pruned dataset."""
    deleted = []

    if PRUNED_DATASET.exists():
        pruned_columns = set(
            pd.read_csv(PRUNED_DATASET, nrows=0).columns
        )

        deleted = [
            col for col in DRUG_FEATURES
            if col in original_columns and col not in pruned_columns
        ]

    elif PREVIOUS_REPORT.exists():
        report = pd.read_csv(PREVIOUS_REPORT)

        feature_col = (
            "feature" if "feature" in report.columns else None
        )

        decision_col = next(
            (
                col for col in ["removal_decision", "decision"]
                if col in report.columns
            ),
            None,
        )

        if feature_col is None or decision_col is None:
            raise ValueError(
                "Could not identify removed features in the previous report."
            )

        decisions = (
            report[decision_col]
            .astype("string")
            .str.strip()
            .str.upper()
        )

        deleted = report.loc[
            decisions.eq("REMOVE"), feature_col
        ].tolist()

        deleted = [
            col for col in deleted
            if col in original_columns and col in DRUG_FEATURES
        ]

    else:
        raise FileNotFoundError(
            "Could not find the pruned dataset or the previous "
            "drug-feature report. Run analyze_drug_features.py first."
        )

    return deleted


def normalize_status(series):
    status = series.astype("string").str.strip().str.casefold()
    return status.mask(status.isin(["", "?"]), pd.NA)


def rate(target, mask):
    """Return group size, early-readmission count and rate in percent."""
    selected = target[mask].dropna()
    n = int(len(selected))

    if n == 0:
        return 0, 0, np.nan

    positives = int(selected.sum())
    return n, positives, positives / n * 100


def calculate_p_value(no_n, no_positive, other_n, other_positive):
    """Fisher exact test, suitable for small comparison groups."""
    if no_n == 0 or other_n == 0:
        return np.nan

    table = np.array([
        [other_positive, other_n - other_positive],
        [no_positive, no_n - no_positive],
    ])

    # Need both target outcomes represented in the data.
    if (table.sum(axis=0) == 0).any():
        return np.nan

    _, p_value = fisher_exact(table, alternative="two-sided")
    return float(p_value)


def adjust_p_values_bh(p_values):
    """Benjamini-Hochberg adjustment for multiple comparisons."""
    result = [np.nan] * len(p_values)

    valid = [
        (i, float(p))
        for i, p in enumerate(p_values)
        if pd.notna(p) and np.isfinite(p)
    ]

    valid.sort(key=lambda item: item[1])
    m = len(valid)
    running_min = 1.0

    for rank in range(m, 0, -1):
        index, p_value = valid[rank - 1]
        running_min = min(running_min, p_value * m / rank)
        result[index] = min(1.0, running_min)

    return result


def main():
    source_path = find_original_dataset()

    print(f"Original dataset: {source_path}")

    original = pd.read_csv(
        source_path,
        dtype={"patient_nbr": "string"},
        low_memory=False,
    )

    target = get_target(original)
    deleted_features = get_deleted_features(original.columns)

    print(f"Original rows: {len(original):,}")
    print(f"Deleted drug features found: {len(deleted_features)}")

    if not deleted_features:
        print(
            "\nNo deleted drug features were found by comparing "
            "the original and pruned dataset columns."
        )
        print("Check that the correct pruned dataset was generated.")
        return

    summary_rows = []
    detail_rows = []

    for feature in deleted_features:
        status = normalize_status(original[feature])

        valid = status.notna() & target.notna()
        no_mask = valid & status.eq("no")
        non_no_mask = valid & status.ne("no")

        no_n, no_positive, no_rate = rate(target, no_mask)
        other_n, other_positive, other_rate = rate(
            target, non_no_mask
        )

        difference = (
            other_rate - no_rate
            if pd.notna(other_rate) and pd.notna(no_rate)
            else np.nan
        )

        risk_ratio = (
            other_rate / no_rate
            if pd.notna(other_rate) and no_rate > 0
            else np.nan
        )

        p_value = calculate_p_value(
            no_n, no_positive, other_n, other_positive
        )

        enough_data = (
            no_n >= MIN_GROUP_COUNT
            and other_n >= MIN_GROUP_COUNT
        )

        if other_n == 0:
            initial_flag = "NO NON-NO ENCOUNTERS - cannot compare"
        elif not enough_data:
            initial_flag = "LOW SAMPLE SIZE - interpret cautiously"
        elif pd.isna(difference):
            initial_flag = "INSUFFICIENT DATA FOR RATE COMPARISON"
        elif abs(difference) >= MIN_DIFFERENCE_PP:
            initial_flag = "LARGE RATE DIFFERENCE - investigate"
        else:
            initial_flag = "No large raw rate difference"

        summary_rows.append({
            "drug_feature": feature,
            "No_encounters": no_n,
            "No_early_readmissions": no_positive,
            "No_readmission_rate_pct": no_rate,
            "non_No_encounters": other_n,
            "non_No_early_readmissions": other_positive,
            "non_No_readmission_rate_pct": other_rate,
            "difference_percentage_points": difference,
            "risk_ratio_non_No_vs_No": risk_ratio,
            "fisher_exact_p_value": p_value,
            "sufficient_group_size": enough_data,
            "initial_flag": initial_flag,
        })

        # Detailed rates for No, Steady, Up, Down, and any
        # unexpected statuses that may be present in the data.
        observed_statuses = sorted(
            status[valid].dropna().unique().tolist()
        )

        for current_status in observed_statuses:
            current_mask = valid & status.eq(current_status)
            n, positives, readmission_rate = rate(
                target, current_mask
            )

            status_difference = (
                readmission_rate - no_rate
                if pd.notna(readmission_rate) and pd.notna(no_rate)
                else np.nan
            )

            detail_rows.append({
                "drug_feature": feature,
                "medication_status": current_status,
                "encounters": n,
                "early_readmissions": positives,
                "readmission_rate_pct": readmission_rate,
                "difference_vs_No_percentage_points": status_difference,
            })

        # An aggregated non-No row makes the primary comparison explicit.
        detail_rows.append({
            "drug_feature": feature,
            "medication_status": "ALL_NON_NO",
            "encounters": other_n,
            "early_readmissions": other_positive,
            "readmission_rate_pct": other_rate,
            "difference_vs_No_percentage_points": difference,
        })

    summary = pd.DataFrame(summary_rows)

    summary["adjusted_p_value_BH"] = adjust_p_values_bh(
        summary["fisher_exact_p_value"].tolist()
    )

    def final_flag(row):
        if row["non_No_encounters"] == 0:
            return "NO COMPARISON POSSIBLE - no non-No encounters"

        if not row["sufficient_group_size"]:
            return "LOW SAMPLE SIZE - rates are unstable"

        diff = row["difference_percentage_points"]
        q_value = row["adjusted_p_value_BH"]

        if pd.isna(diff) or pd.isna(q_value):
            return "INCONCLUSIVE - statistical test unavailable"

        large_difference = abs(diff) >= MIN_DIFFERENCE_PP
        statistically_supported = q_value < 0.05

        if large_difference and statistically_supported:
            return "POTENTIAL ANOMALY - large and statistically supported"
        if large_difference:
            return "LARGE DIFFERENCE - not statistically established"
        if statistically_supported:
            return "STATISTICAL DIFFERENCE - below effect threshold"

        return "NO STRONG ANOMALY DETECTED"

    summary["verification_result"] = summary.apply(final_flag, axis=1)

    detail = pd.DataFrame(detail_rows)

    summary.to_csv(SUMMARY_OUTPUT, index=False)
    detail.to_csv(DETAIL_OUTPUT, index=False)

    display_columns = [
        "drug_feature",
        "No_encounters",
        "No_readmission_rate_pct",
        "non_No_encounters",
        "non_No_readmission_rate_pct",
        "difference_percentage_points",
        "adjusted_p_value_BH",
        "verification_result",
    ]

    print("\n========== DELETED DRUG VERIFICATION ==========")
    print(summary[display_columns].to_string(index=False))

    flagged = summary[
        summary["verification_result"].str.startswith(
            ("POTENTIAL ANOMALY", "LARGE DIFFERENCE")
        )
    ]

    print("\n========== FEATURES TO INVESTIGATE ==========")
    if flagged.empty:
        print(
            "No feature met the current criteria for a large "
            "readmission-rate difference with adequate group support."
        )
    else:
        print(
            flagged[display_columns].to_string(index=False)
        )

    print("\nReports saved:")
    print(f"Summary: {SUMMARY_OUTPUT}")
    print(f"Status-level details: {DETAIL_OUTPUT}")
    print("\nNo dataset rows or columns were changed.")


if __name__ == "__main__":
    main()
