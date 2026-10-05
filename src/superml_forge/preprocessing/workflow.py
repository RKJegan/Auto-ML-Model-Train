"""Preprocessing workflow – orchestrate data preprocessing."""
from __future__ import annotations

import pandas as pd

from .pipeline_builder import build_preprocessor
from .preprocessing_result import PreprocessingResult


def run_preprocessing(X: pd.DataFrame) -> PreprocessingResult:
    """Run preprocessing on the feature DataFrame."""
    preprocessor, numeric_features, categorical_features = build_preprocessor(X)
    preprocessor.fit(X)

    return PreprocessingResult(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        n_features_out=len(numeric_features) + len(categorical_features),
    )
