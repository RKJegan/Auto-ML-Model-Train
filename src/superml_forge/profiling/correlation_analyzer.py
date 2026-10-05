"""Correlation analysis (scaffold)."""
from __future__ import annotations

import pandas as pd


def compute_correlations(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Return the correlation matrix for numeric columns."""
    numeric_df = df.select_dtypes(include="number")
    return numeric_df.corr(method=method)
