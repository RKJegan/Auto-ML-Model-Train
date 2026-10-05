"""Anomaly detection evaluation metrics."""
from __future__ import annotations

from typing import Dict

import numpy as np


def compute_anomaly_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """Compute anomaly detection summary metrics."""
    n_total = len(labels)
    n_anomalies = int(np.sum(labels == -1))

    return {
        "n_total": n_total,
        "n_anomalies": n_anomalies,
        "n_normal": n_total - n_anomalies,
        "anomaly_pct": round(n_anomalies / max(n_total, 1) * 100, 2),
        "mean_score": float(np.mean(scores)) if len(scores) > 0 else 0.0,
        "std_score": float(np.std(scores)) if len(scores) > 0 else 0.0,
    }
