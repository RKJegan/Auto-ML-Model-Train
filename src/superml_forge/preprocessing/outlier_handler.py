"""Outlier detection and handling (scaffold)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def clip_outliers(df: pd.DataFrame, columns: list | None = None, n_std: float = 3.0) -> pd.DataFrame:
    """Clip values beyond n_std standard deviations from the mean."""
    result = df.copy()
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    for col in cols:
        mean = result[col].mean()
        std = result[col].std()
        result[col] = result[col].clip(mean - n_std * std, mean + n_std * std)
    return result
