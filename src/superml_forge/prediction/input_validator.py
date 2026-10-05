"""Input validation and normalization for prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_blank_values(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and convert blank strings to NaN."""
    cleaned = df.copy()
    for col in cleaned.columns:
        series = cleaned[col]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            series = series.astype("string").str.strip()
            series = series.replace(r"^\s*$", np.nan, regex=True)
        cleaned[col] = series
    return cleaned
