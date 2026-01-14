"""Patient metadata extraction from Excel files.

Parses sex and age from various column/key-value layouts in metadata.xlsx files.
"""

import pandas as pd
from pathlib import Path
import numpy as np


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized column names (strip + lowercase)."""
    normalized = {c: str(c).strip().lower() for c in df.columns}
    return df.rename(columns=normalized)


def _get_first_non_null(series: pd.Series):
    """Return first non-null value, or None."""
    if series is None:
        return None
    non_null = series.dropna()
    if non_null.empty:
        return None
    return non_null.iloc[0]


def _try_parse_age_years(value):
    """Parse age from common representations; returns float or None."""
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        if np.isnan(value):
            return None
        return float(value)
    # Handle strings like "45", "45.0", "45 years"
    try:
        text = str(value).strip()
        if text == "" or text.lower() in {"nan", "none"}:
            return None
        # Extract leading numeric token
        token = "".join(ch for ch in text if (ch.isdigit() or ch in ".,"))
        token = token.replace(",", ".")
        return float(token) if token else None
    except Exception:
        return None


def _try_compute_age_from_dates(ct_date_value, born_value):
    """Compute age in years from CT date and birth date; returns float or None."""
    ct_dt = pd.to_datetime(ct_date_value, errors="coerce")
    born_dt = pd.to_datetime(born_value, errors="coerce")
    if pd.isna(ct_dt) or pd.isna(born_dt):
        return None
    return float((ct_dt - born_dt).days) / 365.25


def _extract_sex_and_age(metadata: pd.DataFrame) -> tuple[object | None, float | None]:
    """Extract (sex, age_years) from a metadata table.

    Supports:
    - column layout with headers like Sex/Age (any case/whitespace)
    - column layout where age is computed from CT date + born/dob
    - key/value layout with two columns (e.g. first column = field name)
    """
    df = _normalize_columns(metadata)

    # 1) Header-style columns
    sex_keys = ("sex", "gender")
    age_keys = ("age", "age (years)", "age_years", "age [years]")
    ct_date_keys = ("ct date", "ct_date", "ctdate", "scan date", "scan_date")
    born_keys = ("born", "birth", "date of birth", "dob", "birthdate", "birth_date")

    sex = None
    for key in sex_keys:
        if key in df.columns:
            sex = _get_first_non_null(df[key])
            break

    age_years = None
    for key in age_keys:
        if key in df.columns:
            age_years = _try_parse_age_years(_get_first_non_null(df[key]))
            break

    if age_years is None:
        ct_val = None
        born_val = None
        for key in ct_date_keys:
            if key in df.columns:
                ct_val = _get_first_non_null(df[key])
                break
        for key in born_keys:
            if key in df.columns:
                born_val = _get_first_non_null(df[key])
                break
        age_years = _try_compute_age_from_dates(ct_val, born_val)

    if sex is not None or age_years is not None:
        return sex, age_years

    # 2) Key/value layout fallback (two columns: key, value)
    if df.shape[1] >= 2:
        key_col = df.columns[0]
        val_col = df.columns[1]
        keys = df[key_col].astype(str).str.strip().str.lower()
        values = df[val_col]
        mapping = dict(zip(keys, values))
        sex = mapping.get("sex", mapping.get("gender"))
        age_years = _try_parse_age_years(mapping.get("age"))
        if age_years is None:
            ct_val = mapping.get("ct date", mapping.get("ct_date", mapping.get("ctdate")))
            born_val = mapping.get("born", mapping.get("dob", mapping.get("date of birth")))
            age_years = _try_compute_age_from_dates(ct_val, born_val)
        return sex, age_years

    return None, None

def collect_patient_info(root_directory, metadata_file='metadata.xlsx'):
    """Collect (sex, age) from metadata files in patient subdirectories.

    Args:
        root_directory: Path containing patient folders.
        metadata_file: Metadata filename in each folder.

    Returns:
        Dict mapping patient_id → {'sex': ..., 'age': ...}.
    """
    patient_data = {}  # Dictionary to store results

    for patient_folder in Path(root_directory).iterdir():
        if patient_folder.is_dir():
            metadata_path = patient_folder / metadata_file

            if metadata_path.exists():
                try:
                    # Some metadata workbooks store the actual data in a non-first sheet.
                    metadata = pd.read_excel(metadata_path, sheet_name=None)
                except Exception as exc:
                    print(f"Warning: Failed to read metadata file {metadata_path}: {exc}")
                    continue

                patient_id = patient_folder.name  # Use folder name as ID
                sex = None
                age_years = None
                columns_hint = None

                if isinstance(metadata, dict):
                    # Scan all sheets and use the first one that yields both fields.
                    for sheet_name, sheet_df in metadata.items():
                        if sheet_df is None or sheet_df.empty:
                            continue
                        candidate_sex, candidate_age = _extract_sex_and_age(sheet_df)
                        if candidate_sex is not None and candidate_age is not None:
                            sex, age_years = candidate_sex, candidate_age
                            break
                        if columns_hint is None:
                            columns_hint = (sheet_name, ", ".join(str(c) for c in sheet_df.columns))
                else:
                    if metadata.empty:
                        print(f"Warning: Empty metadata file in {metadata_path}")
                        continue
                    sex, age_years = _extract_sex_and_age(metadata)
                    columns_hint = ("<default>", ", ".join(str(c) for c in metadata.columns))

                if sex is None or age_years is None:
                    if columns_hint is None:
                        sheet_part = ""
                        cols_part = "<no non-empty sheets>"
                    else:
                        sheet_part = f" (example sheet: {columns_hint[0]})"
                        cols_part = columns_hint[1]
                    print(
                        "Warning: Missing required patient metadata "
                        f"(sex={sex is not None}, age={age_years is not None}) in {metadata_path}{sheet_part}. "
                        f"Found columns: [{cols_part}]"
                    )
                    continue

                patient_data[patient_id] = {'sex': sex, 'age': age_years}

    return patient_data

def get_as_numpy(patient_info, item):
    """Extract `item` ('sex' or 'age') from patient_info as NumPy array."""
    
    return np.array([info[item] for info in patient_info.values()])

def get_patientid_as_numpy(patient_info):
    """Return patient IDs from patient_info as NumPy array."""
    return np.array(list(patient_info.keys()))

if __name__ == '__main__':
    database_directory = '/mnt/database/IOR_femurs/raw'
    patient_info = collect_patient_info(database_directory)
    sex = get_as_numpy(patient_info, 'sex')
    age = get_as_numpy(patient_info, 'age')
    patient_ids = get_patientid_as_numpy(patient_info)