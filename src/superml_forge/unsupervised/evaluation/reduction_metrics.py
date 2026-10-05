"""Dimensionality reduction evaluation metrics."""
from __future__ import annotations

from typing import Dict

import numpy as np


def compute_reduction_metrics(model, X_original: np.ndarray) -> Dict[str, float]:
    """Compute reduction quality metrics."""
    metrics = {}

    if hasattr(model, "explained_variance_ratio_"):
        evr = model.explained_variance_ratio_
        metrics["total_explained_variance"] = float(np.sum(evr))
        for i, v in enumerate(evr):
            metrics[f"PC{i+1}_variance"] = float(v)

    return metrics
