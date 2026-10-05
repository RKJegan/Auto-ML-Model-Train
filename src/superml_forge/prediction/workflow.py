"""Prediction workflow."""
from __future__ import annotations

from typing import List

import pandas as pd

from .predictor import predict_with_input
from .prediction_result import PredictionResult


def run_prediction(model, input_df: pd.DataFrame, feature_columns: List[str]) -> PredictionResult:
    """Run the prediction workflow."""
    predictions = predict_with_input(model, input_df, feature_columns)
    return PredictionResult(
        predictions=predictions,
        feature_columns=feature_columns,
    )
