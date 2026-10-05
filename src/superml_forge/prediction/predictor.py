"""Core prediction logic."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .input_validator import normalize_blank_values


def align_features(input_df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """Align input columns to match the expected feature order."""
    aligned = input_df.copy()
    aligned.columns = aligned.columns.astype(str)

    for col in feature_columns:
        if col not in aligned.columns:
            aligned[col] = np.nan
    aligned = aligned[feature_columns]
    return aligned


def predict_with_input(model, input_df: pd.DataFrame, feature_columns: List[str]):
    """Run prediction on a single input DataFrame."""
    if input_df.empty:
        raise ValueError("Input data for prediction is empty.")
    if not feature_columns:
        raise ValueError("Feature metadata is missing. Please retrain the model.")

    cleaned = normalize_blank_values(input_df)
    aligned = align_features(cleaned, feature_columns)
    prediction = model.predict(aligned)
    return prediction
