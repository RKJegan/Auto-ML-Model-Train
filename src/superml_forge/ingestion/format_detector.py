"""Detect file format and list Excel sheets."""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".tsv", ".json", ".parquet", ".txt"}


def detect_extension(file_name: str) -> str:
    """Return the lowercase file extension from *file_name*."""
    ext = Path(file_name).suffix.lower()
    if not ext:
        raise ValueError("Could not detect file type. Please upload a file with a valid extension.")
    return ext


def list_excel_sheets(uploaded_file) -> List[str]:
    """Return the list of sheet names from an Excel workbook."""
    try:
        uploaded_file.seek(0)
        excel_file = pd.ExcelFile(uploaded_file)
        return excel_file.sheet_names
    except Exception as exc:
        raise ValueError(f"Unable to read Excel sheets: {exc}") from exc
