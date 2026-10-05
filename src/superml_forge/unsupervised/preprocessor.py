"""Preprocessing for unsupervised learning – numeric-only impute + scale."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_for_unsupervised(
    df: pd.DataFrame,
    columns: List[str] | None = None,
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Prepare data for unsupervised learning.

    - Selects only specified columns (or all if None).
    - Encodes categorical columns with LabelEncoder.
    - Imputes missing values (median for numeric, mode for encoded categorical).
    - Scales all features with StandardScaler.

    Returns:
        X_scaled: numpy array of preprocessed features
        feature_names: list of column names used
        df_clean: the cleaned DataFrame before scaling (for reference)
    """
    if columns:
        work_df = df[columns].copy()
    else:
        work_df = df.copy()

    feature_names = work_df.columns.tolist()

    # Encode categorical columns
    label_encoders = {}
    for col in work_df.columns:
        if work_df[col].dtype == "object" or pd.api.types.is_string_dtype(work_df[col]):
            le = LabelEncoder()
            # Handle NaN: fill temporarily, encode, then restore NaN positions
            mask = work_df[col].isna()
            work_df[col] = work_df[col].fillna("__MISSING__")
            work_df[col] = le.fit_transform(work_df[col].astype(str))
            work_df.loc[mask, col] = np.nan
            label_encoders[col] = le

    # Convert to numeric
    work_df = work_df.apply(pd.to_numeric, errors="coerce")
    df_clean = work_df.copy()

    # Impute + Scale
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    X_scaled = pipeline.fit_transform(work_df)
    return X_scaled, feature_names, df_clean
