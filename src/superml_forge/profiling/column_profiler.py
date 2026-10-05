"""Per-column statistical profiling (scaffold)."""
from __future__ import annotations

import pandas as pd


def profile_column(series: pd.Series) -> dict:
    """Return summary statistics for a single column."""
    stats = {
        "dtype": str(series.dtype),
        "count": int(series.count()),
        "missing": int(series.isna().sum()),
        "unique": int(series.nunique()),
    }
    if pd.api.types.is_numeric_dtype(series):
        stats.update({
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
        })
    return stats
