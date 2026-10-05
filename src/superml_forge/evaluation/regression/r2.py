"""R-squared metric."""
from __future__ import annotations

from sklearn.metrics import r2_score


def compute(y_true, y_pred) -> float:
    """Compute R²."""
    return float(r2_score(y_true, y_pred))
