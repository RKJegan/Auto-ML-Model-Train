"""Duplicate row handling (scaffold)."""
from __future__ import annotations

from typing import Literal

import pandas as pd


def remove_duplicates(df: pd.DataFrame, keep: Literal["first", "last"] | bool = "first") -> pd.DataFrame:
    """Remove duplicate rows from the DataFrame."""
    return df.drop_duplicates(keep=keep).reset_index(drop=True)
