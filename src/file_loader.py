from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".tsv", ".json", ".parquet", ".txt"}


def detect_extension(file_name: str) -> str:
    ext = Path(file_name).suffix.lower()
    if not ext:
        raise ValueError("Could not detect file type. Please upload a file with a valid extension.")
    return ext


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = (
        cleaned.columns.astype(str).str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    )
    return cleaned


def list_excel_sheets(uploaded_file) -> List[str]:
    try:
        uploaded_file.seek(0)
        excel_file = pd.ExcelFile(uploaded_file)
        return excel_file.sheet_names
    except Exception as exc:
        raise ValueError(f"Unable to read Excel sheets: {exc}") from exc


def load_data(uploaded_file, sheet_name: Optional[str] = None) -> pd.DataFrame:
    if uploaded_file is None:
        raise ValueError("No file was uploaded.")

    ext = detect_extension(uploaded_file.name)
    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file format '{ext}'. Supported formats: {supported}")

    try:
        uploaded_file.seek(0)
        if ext == ".csv":
            df = pd.read_csv(uploaded_file)
        elif ext in {".xlsx", ".xls"}:
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name or 0)
        elif ext == ".tsv":
            df = pd.read_csv(uploaded_file, sep="\t")
        elif ext == ".json":
            # pandas handles common JSON tabular structures.
            df = pd.read_json(uploaded_file)
        elif ext == ".parquet":
            df = pd.read_parquet(uploaded_file)
        else:
            # .txt support is optional. Try delimiter inference first.
            df = pd.read_csv(uploaded_file, sep=None, engine="python")
    except Exception as exc:
        raise ValueError(
            f"Failed to read '{uploaded_file.name}'. The file may be corrupted or not truly tabular. Details: {exc}"
        ) from exc

    if not isinstance(df, pd.DataFrame):
        raise ValueError("The uploaded content could not be converted into a tabular DataFrame.")

    if df.empty:
        raise ValueError("Uploaded data is empty after parsing.")

    return standardize_column_names(df)
