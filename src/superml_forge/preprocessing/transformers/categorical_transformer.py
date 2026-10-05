"""Categorical feature transformer pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


def _clean_categorical_values(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize string values and convert blanks to NaN for categorical columns."""
    cleaned = data.copy()
    for col in cleaned.columns:
        cleaned[col] = cleaned[col].astype(object)
        mask = cleaned[col].notna()
        cleaned.loc[mask, col] = cleaned.loc[mask, col].astype(str).str.strip()
        cleaned[col] = cleaned[col].replace(r"^\s*$", np.nan, regex=True)
        
    cleaned = cleaned.fillna(np.nan)
    cleaned = cleaned.replace({pd.NA: np.nan})
    return cleaned


def build_categorical_pipeline(
    imputer_strategy: str = "most_frequent",
    fill_value: str = "missing",
    handle_unknown: Literal["error", "ignore", "infrequent_if_exist"] = "ignore",
) -> Pipeline:
    """Build a pipeline that cleans, imputes, and one-hot encodes categorical features."""
    return Pipeline(
        steps=[
            ("cleaner", FunctionTransformer(_clean_categorical_values, validate=False)),
            ("imputer", SimpleImputer(strategy=imputer_strategy, fill_value=fill_value)),
            ("onehot", OneHotEncoder(handle_unknown=handle_unknown)),
        ]
    )
