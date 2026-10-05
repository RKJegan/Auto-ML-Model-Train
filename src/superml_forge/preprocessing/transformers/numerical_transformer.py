"""Numeric feature transformer pipeline."""
from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_numeric_pipeline(
    imputer_strategy: str = "median",
) -> Pipeline:
    """Build a pipeline that imputes and scales numeric features."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=imputer_strategy)),
            ("scaler", StandardScaler()),
        ]
    )
