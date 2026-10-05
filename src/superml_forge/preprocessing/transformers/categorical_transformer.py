"""Categorical feature transformer pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


def _clean_categorical_values(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize string values and convert blanks to NaN for categorical columns."""
    cleaned = data.copy()
    for col in cleaned.columns:
        series = cleaned[col]
        series = series.astype("string").str.strip()
        series = series.replace(r"^\s*$", np.nan, regex=True)
        cleaned[col] = series
    return cleaned


def build_categorical_pipeline(
    imputer_strategy: str = "constant",
    fill_value: str = "missing",
    handle_unknown: str = "ignore",
) -> Pipeline:
    """Build a pipeline that cleans, imputes, and one-hot encodes categorical features."""
    return Pipeline(
        steps=[
            ("cleaner", FunctionTransformer(_clean_categorical_values, validate=False)),
            ("imputer", SimpleImputer(strategy=imputer_strategy, fill_value=fill_value)),
            ("onehot", OneHotEncoder(handle_unknown=handle_unknown)),
        ]
    )
