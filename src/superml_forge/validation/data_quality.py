"""Data sanitization utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from string columns and convert blank strings to NaN."""
    cleaned = df.copy()
    for col in cleaned.columns:
        if pd.api.types.is_object_dtype(cleaned[col]) or pd.api.types.is_string_dtype(cleaned[col]):
            cleaned[col] = cleaned[col].astype("string").str.strip()
            cleaned[col] = cleaned[col].replace(r"^\s*$", np.nan, regex=True)
    return cleaned
