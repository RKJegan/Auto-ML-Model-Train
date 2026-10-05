"""Accuracy metric."""
from __future__ import annotations

from sklearn.metrics import accuracy_score


def compute(y_true, y_pred, **kwargs) -> float:
    """Compute accuracy."""
    return float(accuracy_score(y_true, y_pred, **kwargs))
