"""Mean Squared Error metric."""
from __future__ import annotations

from sklearn.metrics import mean_squared_error


def compute(y_true, y_pred) -> float:
    """Compute MSE."""
    return float(mean_squared_error(y_true, y_pred))
