"""Core logic for detecting classification vs regression tasks."""
from __future__ import annotations

import pandas as pd


def detect_problem_type(target: pd.Series) -> str:
    """
    Infer whether the problem is classification or regression.

    Heuristic:
    - If the target dtype is object/category/bool, treat as classification.
    - Otherwise, if the number of unique values is relatively small
      compared to the dataset size, treat as classification.
    - Else, treat as regression.
    """
    if target.dtype == "O" or isinstance(target.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(
        target
    ):
        return "classification"

    # Numeric target: decide based on number of unique values
    n_unique = target.nunique(dropna=True)
    n_total = len(target)
    unique_ratio = n_unique / max(n_total, 1)

    if n_unique <= 20 or unique_ratio < 0.05:
        return "classification"
    return "regression"
