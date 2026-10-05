"""Mean Absolute Percentage Error metric (scaffold)."""
from __future__ import annotations

import numpy as np


def compute(y_true, y_pred) -> float:
    """Compute MAPE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return float("inf")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
