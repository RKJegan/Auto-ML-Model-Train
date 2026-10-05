"""Clustering evaluation metrics."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def compute_clustering_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute standard clustering quality metrics."""
    unique_labels = set(labels)
    unique_labels.discard(-1)
    n_clusters = len(unique_labels)

    if n_clusters < 2 or n_clusters >= len(X):
        return {"n_clusters": n_clusters}

    mask = labels != -1
    if mask.sum() < 2:
        return {"n_clusters": n_clusters}

    metrics = {"n_clusters": n_clusters}
    try:
        metrics["silhouette_score"] = float(silhouette_score(X[mask], labels[mask]))
    except Exception:
        pass
    try:
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(X[mask], labels[mask]))
    except Exception:
        pass
    try:
        metrics["davies_bouldin"] = float(davies_bouldin_score(X[mask], labels[mask]))
    except Exception:
        pass

    return metrics
