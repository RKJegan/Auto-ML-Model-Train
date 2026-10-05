"""Data sanitization utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data and ensure compatibility with scikit-learn."""
    cleaned = df.copy()
    
    # First convert pandas extended dtypes (Int64, string, etc) to standard numpy dtypes
    for col in cleaned.columns:
        if isinstance(cleaned[col].dtype, pd.core.dtypes.dtypes.ExtensionDtype) or pd.api.types.is_string_dtype(cleaned[col]):
            if pd.api.types.is_numeric_dtype(cleaned[col]):
                cleaned[col] = cleaned[col].astype("float64")
            else:
                cleaned[col] = cleaned[col].astype(object)

    # Now handle string stripping
    for col in cleaned.columns:
        if pd.api.types.is_object_dtype(cleaned[col]):
            mask = cleaned[col].notna()
            cleaned.loc[mask, col] = cleaned.loc[mask, col].astype(str).str.strip()
            cleaned[col] = cleaned[col].replace(r"^\s*$", np.nan, regex=True)

    # Finally replace all pd.NA with np.nan
    cleaned = cleaned.fillna(np.nan)
    cleaned = cleaned.replace({pd.NA: np.nan})
    return cleaned
