"""Confusion matrix computation."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix


def compute(y_true, y_pred, labels=None) -> np.ndarray:
    """Compute the confusion matrix."""
    return confusion_matrix(y_true, y_pred, labels=labels)
