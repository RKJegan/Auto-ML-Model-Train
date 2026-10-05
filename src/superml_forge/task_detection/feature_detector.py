"""Feature type detection (scaffold)."""
from __future__ import annotations



import pandas as pd


def detect_feature_types(df: pd.DataFrame) -> dict:
    """Classify columns as numeric, categorical, datetime, or text."""
    import numpy as np
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]
    return {"numeric": numeric, "categorical": categorical}
