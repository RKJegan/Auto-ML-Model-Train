"""Distribution analysis (scaffold)."""
from __future__ import annotations

import pandas as pd


def analyze_distribution(series: pd.Series) -> dict:
    """Analyze the distribution of a numeric series."""
    if not pd.api.types.is_numeric_dtype(series):
        return {"type": "categorical", "n_unique": int(series.nunique())}
    return {
        "type": "numeric",
        "skew": float(series.skew()),
        "kurtosis": float(series.kurtosis()),
    }
