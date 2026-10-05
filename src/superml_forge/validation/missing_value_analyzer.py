"""Analyse missing values in a DataFrame."""
from __future__ import annotations

import pandas as pd


def get_missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame summarising columns with missing values.

    Columns: missing_count, missing_pct.  Only rows with missing_count > 0 are included.
    """
    null_counts = df.isna().sum()
    summary = pd.DataFrame(
        {
            "missing_count": null_counts,
            "missing_pct": ((null_counts / len(df)) * 100).round(2),
        }
    )
    return summary[summary["missing_count"] > 0].sort_values("missing_count", ascending=False)
