"""Root Mean Squared Error metric."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error


def compute(y_true, y_pred) -> float:
    """Compute RMSE."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))
