"""Recall metric."""
from __future__ import annotations

from sklearn.metrics import recall_score


def compute(y_true, y_pred, **kwargs) -> float:
    """Compute recall."""
    return float(recall_score(y_true, y_pred, **kwargs))
