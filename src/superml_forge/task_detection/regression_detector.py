"""Regression task detection heuristics (scaffold)."""
from __future__ import annotations

import pandas as pd


def is_regression_target(target: pd.Series) -> bool:
    """Return True if the target looks like a regression target."""
    if not pd.api.types.is_numeric_dtype(target):
        return False
    n_unique = target.nunique(dropna=True)
    return n_unique > 20 and (n_unique / max(len(target), 1)) >= 0.05
