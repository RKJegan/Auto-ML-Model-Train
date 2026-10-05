from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def _normalize_blank_values(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for col in cleaned.columns:
        series = cleaned[col]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            series = series.astype("string").str.strip()
            series = series.replace(r"^\s*$", np.nan, regex=True)
        cleaned[col] = series
    return cleaned


def align_features(input_df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    aligned = input_df.copy()
    aligned.columns = aligned.columns.astype(str)

    # Add missing columns as NaN, then enforce exact order.
    for col in feature_columns:
        if col not in aligned.columns:
            aligned[col] = np.nan
    aligned = aligned[feature_columns]
    return aligned


def predict_with_input(model, input_df: pd.DataFrame, feature_columns: List[str]):
    if input_df.empty:
        raise ValueError("Input data for prediction is empty.")
    if not feature_columns:
        raise ValueError("Feature metadata is missing. Please retrain the model.")

    cleaned = _normalize_blank_values(input_df)
    aligned = align_features(cleaned, feature_columns)
    prediction = model.predict(aligned)
    return prediction
