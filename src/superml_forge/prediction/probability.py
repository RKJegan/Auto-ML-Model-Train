"""Prediction probability utilities (scaffold)."""
from __future__ import annotations

import pandas as pd


def get_prediction_probabilities(model, input_df: pd.DataFrame) -> pd.DataFrame | None:
    """Return class probabilities if the model supports predict_proba."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)
        if hasattr(model, "classes_"):
            return pd.DataFrame(proba, columns=model.classes_)
        return pd.DataFrame(proba)
    return None
