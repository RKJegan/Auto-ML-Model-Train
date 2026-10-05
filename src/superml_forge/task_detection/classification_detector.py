"""Classification task detection heuristics (scaffold)."""
from __future__ import annotations

import pandas as pd


def is_classification_target(target: pd.Series, max_unique: int = 20) -> bool:
    """Return True if the target looks like a classification target."""
    if target.dtype == "O" or isinstance(target.dtype, pd.CategoricalDtype):
        return True
    return target.nunique(dropna=True) <= max_unique
