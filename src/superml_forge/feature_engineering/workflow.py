"""Feature engineering workflow (scaffold)."""
from __future__ import annotations

import pandas as pd

from .feature_result import FeatureResult


def run_feature_engineering(X: pd.DataFrame) -> FeatureResult:
    """Run feature engineering – currently a pass-through."""
    return FeatureResult(
        selected_features=X.columns.tolist(),
        n_features_in=len(X.columns),
        n_features_out=len(X.columns),
    )
