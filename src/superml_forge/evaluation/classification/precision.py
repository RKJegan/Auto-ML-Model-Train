"""Precision metric."""
from __future__ import annotations

from sklearn.metrics import precision_score


def compute(y_true, y_pred, **kwargs) -> float:
    """Compute precision."""
    return float(precision_score(y_true, y_pred, **kwargs))
