"""F1-score metric."""
from __future__ import annotations

from sklearn.metrics import f1_score


def compute(y_true, y_pred, **kwargs) -> float:
    """Compute f1 score."""
    return float(f1_score(y_true, y_pred, **kwargs))
