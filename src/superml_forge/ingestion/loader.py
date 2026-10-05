"""Load tabular data from various file formats."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .format_detector import SUPPORTED_EXTENSIONS, detect_extension
from .file_handler import standardize_column_names


def load_data(uploaded_file, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    Read a tabular file into a DataFrame.

    Supports CSV, Excel (.xlsx/.xls), TSV, JSON, Parquet, and plain text.
    """
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
            df = pd.read_json(uploaded_file)
        elif ext == ".parquet":
            df = pd.read_parquet(uploaded_file)
        else:
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
