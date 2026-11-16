"""Build an inventory of every ICD-9-CM code appearing in diag_1, diag_2, diag_3.

Reads the cleaned CSV by default (so it uses the same population as Care Continuum).
Creates diagnosis_code_inventory.csv with frequency counts, a broad project grouping,
and—when internet access is available—official-style code descriptions from the U.S.
National Library of Medicine Clinical Table Search Service.

Run from the Care Continuum project folder:
    python diagnosis_code_inventory.py

Optional:
    python diagnosis_code_inventory.py dataset/diabetic_data.xlsx
    python diagnosis_code_inventory.py dataset/diabetic_data_cleaned.csv --no-lookup
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

DIAG_COLUMNS = ["diag_1", "diag_2", "diag_3"]
API_URL = "https://clinicaltables.nlm.nih.gov/api/icd9cm_dx/v3/search"
CACHE_PATH = Path("diagnosis_description_cache.json")
DEFAULT_OUTPUT = Path("diagnosis_code_inventory.csv")


def resolve_input(explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path

    candidates = [
        Path("dataset/diabetic_data_cleaned.csv"),
        Path("diabetic_data_cleaned.csv"),
        Path("dataset/diabetic_data.xlsx"),
        Path("dataset/diabetic_data.csv"),
        Path("diabetic_data.xlsx"),
        Path("diabetic_data.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find the cleaned CSV or original Excel/CSV. Pass the file path, e.g. "
        "python diagnosis_code_inventory.py dataset/diabetic_data_cleaned.csv"
    )


def load_data(path: Path) -> pd.DataFrame:
    dtypes = {column: "string" for column in DIAG_COLUMNS}
    if path.suffix.lower() in {".xlsx", ".xls", ".xlsm"}:
        df = pd.read_excel(path, dtype=dtypes, keep_default_na=False)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype=dtypes, keep_default_na=False, low_memory=False)
    else:
        raise ValueError("Input must be a .csv, .xlsx, .xls, or .xlsm file.")

    missing_columns = [column for column in DIAG_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Input is missing diagnosis columns: {missing_columns}")
    return df


def normalize_code(value: object) -> str:
    """Normalize representation without treating the code as a numeric quantity.

    In ICD-9-CM, short numeric codes in ranges such as 008 and 038 can lose their
    leading zero when a spreadsheet treats them as numbers. We restore the first
    three numeric characters for the base code, while preserving any suffix.
    Examples: 8 -> 008, 38 -> 038, 250.83 -> 250.83, V27 -> V27.
    """
    if value is None or pd.isna(value):
        return ""
    code = str(value).strip().upper()
    if code in {"", "?", "NAN", "NA", "N/A", "<NA>"}:
        return ""
    code = re.sub(r"\s+", "", code)

    # Only pad numeric diagnosis roots. Keep V/E codes and their suffixes as strings.
    match = re.fullmatch(r"(\d+)(\.\d+)?", code)
    if match:
        root = match.group(1)
        suffix = match.group(2) or ""
        if len(root) < 3:
            root = root.zfill(3)
        code = root + suffix
    return code


def broad_project_group(code: str) -> str:
    """Coarse project grouping; this is not a complete clinical coding system."""
    if not code:
        return "Missing"
    if code.startswith(("V", "E")):
        return "Other (V/E codes)"

    root = code.split(".", 1)[0]
    if not root.isdigit():
        return "Other"
    number = int(root)

    if number == 250:
        return "Diabetes"
    if 140 <= number <= 239:
        return "Neoplasms"
    if 390 <= number <= 459:
        return "Circulatory"
    if 460 <= number <= 519:
        return "Respiratory"
    if 520 <= number <= 579:
        return "Digestive"
    if 580 <= number <= 629:
        return "Genitourinary"
    if 710 <= number <= 739:
        return "Musculoskeletal"
    if 800 <= number <= 999:
        return "Injuries"
    return "Other"


def code_key(code: str) -> str:
    """Comparison key that ignores the printed dot but preserves all digits."""
    return code.replace(".", "").upper()


def load_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in data.items() if v}
    except (OSError, json.JSONDecodeError):
        print(f"Warning: unable to read cache {path}; starting a fresh cache.")
        return {}


def lookup_description(code: str) -> tuple[str, str]:
    """Return (description, status) from NLM's ICD-9-CM diagnosis lookup API."""
    # NLM's `code` field is usually stored without the dot. Searching with that
    # representation and matching the returned dotted code avoids prefix mistakes.
    params = {
        "terms": code_key(code),
        "sf": "code",
        "df": "code_dotted,long_name",
        "count": "100",
        "maxList": "100",
    }
    url = API_URL + "?" + urlencode(params)
    request = Request(url, headers={"User-Agent": "CareContinuumDiagnosisInventory/1.0"})

    try:
        with urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
        # NLM response element 4 (index 3) contains display rows; each row is
        # [code_dotted, long_name] for the requested df fields.
        rows = payload[3] if len(payload) > 3 and payload[3] else []
        for row in rows:
            if len(row) >= 2 and code_key(normalize_code(row[0])) == code_key(code):
                description = str(row[1]).strip()
                if description:
                    return description, "Found in NLM ICD-9-CM lookup"
        return "", "Not found in NLM lookup (check code/codebook)"
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        return "", f"Lookup unavailable: {type(exc).__name__}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory diagnosis codes and look up ICD-9-CM descriptions.")
    parser.add_argument("input", nargs="?", help="CSV or Excel input; default prefers the cleaned dataset.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path (default: diagnosis_code_inventory.csv).")
    parser.add_argument("--no-lookup", action="store_true", help="Skip online description lookups; frequencies and broad groups are still generated.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel online lookups (default: 4).")
    args = parser.parse_args()

    source_path = resolve_input(args.input)
    df = load_data(source_path)
    print(f"Loaded {source_path} | rows={len(df):,}")

    # Normalize diagnosis codes into string values; this does not edit the source file.
    normalized = pd.DataFrame(index=df.index)
    raw_forms: dict[str, set[str]] = defaultdict(set)
    for col in DIAG_COLUMNS:
        normalized[col] = df[col].map(normalize_code)
        for raw_value, canonical in zip(df[col].astype("string"), normalized[col]):
            if canonical:
                raw_forms[canonical].add(str(raw_value).strip())

    codes = sorted(
        set().union(*(set(normalized[col].dropna()) - {""} for col in DIAG_COLUMNS)),
        key=lambda item: (item[0] in {"V", "E"}, item),
    )
    if not codes:
        raise ValueError("No non-missing diagnosis codes were found.")

    rows: list[dict[str, object]] = []
    for code in codes:
        counts = {col: int(normalized[col].eq(code).sum()) for col in DIAG_COLUMNS}
        any_in_row = normalized[DIAG_COLUMNS].eq(code).any(axis=1)
        rows.append({
            "diagnosis_code": code,
            "diag_1_count": counts["diag_1"],
            "diag_2_count": counts["diag_2"],
            "diag_3_count": counts["diag_3"],
            "total_occurrences": sum(counts.values()),
            "encounters_with_code": int(any_in_row.sum()),
            "project_group_preview": broad_project_group(code),
            "raw_forms_seen": " | ".join(sorted(raw_forms[code])),
            "description": "",
            "description_lookup_status": "Not requested" if args.no_lookup else "Pending",
        })

    cache_path = Path(CACHE_PATH)
    cache = load_cache(cache_path)
    todo = [code for code in codes if code not in cache]
    descriptions: dict[str, str] = dict(cache)
    statuses: dict[str, str] = {code: "Found in local cache" for code in cache}

    if not args.no_lookup:
        print(f"Looking up descriptions for {len(todo):,} codes using the NLM ICD-9-CM service.")
        print("This needs internet access. The inventory will still be saved if lookups fail.")
        with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 8))) as executor:
            future_to_code = {executor.submit(lookup_description, code): code for code in todo}
            completed = 0
            for future in as_completed(future_to_code):
                code = future_to_code[future]
                try:
                    description, status = future.result()
                except Exception as exc:  # Defensive: keep inventory creation resilient.
                    description, status = "", f"Lookup error: {type(exc).__name__}"
                statuses[code] = status
                if description:
                    descriptions[code] = description
                completed += 1
                if completed % 50 == 0 or completed == len(todo):
                    print(f"  Description lookup completed: {completed:,}/{len(todo):,}")

        # Cache only successful descriptions, so unavailable lookups can be retried later.
        successful_cache = {code: desc for code, desc in descriptions.items() if desc}
        cache_path.write_text(json.dumps(successful_cache, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Description cache saved to {cache_path.resolve()}")

    for row in rows:
        code = str(row["diagnosis_code"])
        if not args.no_lookup:
            row["description"] = descriptions.get(code, "")
            row["description_lookup_status"] = statuses.get(code, "Not looked up")

    result = pd.DataFrame(rows)
    result.sort_values(["total_occurrences", "diagnosis_code"], ascending=[False, True], inplace=True)
    output_path = Path(args.output)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\nDiagnosis-code inventory complete.")
    print(f"Distinct non-missing codes: {len(codes):,}")
    for col in DIAG_COLUMNS:
        count = int((normalized[col] != "").sum())
        unique = int(normalized.loc[normalized[col] != "", col].nunique())
        print(f"  {col}: {unique:,} distinct codes across {count:,} non-missing values")
    print(f"Output saved to: {output_path.resolve()}")
    print("The source dataset was not changed. These broad groups are a preliminary project mapping, not clinical coding advice.")


if __name__ == "__main__":
    main()
