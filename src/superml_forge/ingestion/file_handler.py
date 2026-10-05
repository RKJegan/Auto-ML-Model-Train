"""Column name standardization and file-level utilities."""
from __future__ import annotations

import pandas as pd


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, strip, and replace whitespace in column names with underscores."""
    cleaned = df.copy()
    cleaned.columns = (
        cleaned.columns.astype(str).str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    )
    return cleaned
