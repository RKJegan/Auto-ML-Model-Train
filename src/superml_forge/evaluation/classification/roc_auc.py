"""ROC AUC metric."""
from __future__ import annotations

from sklearn.metrics import roc_auc_score


def compute(y_true, y_pred, **kwargs) -> float:
    """Compute roc auc."""
    return float(roc_auc_score(y_true, y_pred, **kwargs))
