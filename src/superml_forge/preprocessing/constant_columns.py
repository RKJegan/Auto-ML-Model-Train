"""Constant column removal (scaffold)."""
from __future__ import annotations

from typing import List

import pandas as pd


def find_constant_columns(df: pd.DataFrame) -> List[str]:
    """Return column names that have zero variance."""
    return [col for col in df.columns if df[col].nunique(dropna=True) <= 1]


def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns with zero variance."""
    constant_cols = find_constant_columns(df)
    return df.drop(columns=constant_cols)
