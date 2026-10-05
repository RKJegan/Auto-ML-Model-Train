"""Mean Absolute Error metric."""
from __future__ import annotations

from sklearn.metrics import mean_absolute_error


def compute(y_true, y_pred) -> float:
    """Compute MAE."""
    return float(mean_absolute_error(y_true, y_pred))
