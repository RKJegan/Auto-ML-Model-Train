"""Build sklearn preprocessing pipelines."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from .transformers.numerical_transformer import build_numeric_pipeline
from .transformers.categorical_transformer import build_categorical_pipeline


def build_preprocessor(
    X: pd.DataFrame,
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Create a ColumnTransformer that:
    - Imputes numeric features with median and scales them.
    - Imputes categorical features with most-frequent and one-hot encodes them.

    Returns the preprocessor and the lists of numeric and categorical feature names.
    """
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = build_numeric_pipeline()
    categorical_transformer = build_categorical_pipeline()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


def build_full_pipeline(
    X: pd.DataFrame,
    model,
) -> Pipeline:
    """
    Attach the preprocessing ColumnTransformer in front of a model
    to create a full sklearn Pipeline.
    """
    preprocessor, _, _ = build_preprocessor(X)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline
