"""Duplicate row detection (scaffold)."""
from __future__ import annotations

import pandas as pd


def detect_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Return duplicate rows in the DataFrame."""
    return df[df.duplicated(keep=False)]
