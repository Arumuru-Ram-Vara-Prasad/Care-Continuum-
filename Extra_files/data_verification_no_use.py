import pandas as pd

df = pd.read_excel("dataset/diabetic_data.xlsx", na_values=["?"], keep_default_na=False)

# 1. Repeat patients
visits = df["patient_nbr"].value_counts()
print("Encounters:", len(df))
print("Unique patients:", df["patient_nbr"].nunique())
print("Patients with 2+ encounters:", (visits > 1).sum())
print("Most encounters for one patient:", visits.max())

# 2. Discharge codes (compare with IDS_mapping.csv)
print(df["discharge_disposition_id"].value_counts().sort_index())

# 3. Target balance
print(df["readmitted"].value_counts(normalize=True))

# 4. Check labels for death/hospice discharge codes
exclude_codes = ["11", "19", "20"]

disposition = (
    df["discharge_disposition_id"]
    .astype("string")
    .str.strip()
)

excluded = disposition.isin(exclude_codes)

print("\nReadmission labels by proposed exclusion code:")
print(pd.crosstab(
    disposition[excluded],
    df.loc[excluded, "readmitted"],
    dropna=False
))

# 5. Target distribution after proposed exclusions
clean_df = df.loc[~excluded].copy()

print("\nEncounters remaining:", len(clean_df))
print("\nTarget distribution after exclusions:")
print(clean_df["readmitted"].value_counts())

print("\nTarget percentages after exclusions:")
print(
    (clean_df["readmitted"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\nBinary target balance after exclusions:")
binary_target = clean_df["readmitted"].eq("<30").astype(int)
print(binary_target.value_counts().rename({
    0: "Not readmitted within 30 days",
    1: "Readmitted within 30 days"
}))