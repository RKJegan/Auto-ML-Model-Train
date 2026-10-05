"""Clustering training logic."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from .registry import get_clustering_models


@dataclass
class ClusteringResult:
    """Result of a single clustering model."""
    name: str
    labels: np.ndarray
    n_clusters: int
    metrics: Dict[str, float] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)


def _compute_clustering_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute clustering quality metrics."""
    unique_labels = set(labels)
    # Remove noise label (-1) for metric calculation
    unique_labels.discard(-1)
    n_clusters = len(unique_labels)

    if n_clusters < 2 or n_clusters >= len(X):
        return {"n_clusters": n_clusters}

    # Filter out noise points for metrics
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


def train_clustering_models(
    X: np.ndarray,
    feature_names: List[str],
    selected_model_name: Optional[str] = None,
    manual_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Train clustering models and compute metrics.

    Returns dict with:
        - "results": list of ClusteringResult
        - "best": best ClusteringResult (by silhouette score)
        - "comparison": DataFrame comparing all models
    """
    n_clusters = 3
    if manual_params and "n_clusters" in manual_params:
        n_clusters = manual_params["n_clusters"]
    elif manual_params and "n_components" in manual_params:
        n_clusters = manual_params["n_components"]

    models = get_clustering_models(n_clusters=n_clusters)

    if selected_model_name:
        if selected_model_name not in models:
            raise ValueError(f"Unknown clustering model: {selected_model_name}")
        models = {selected_model_name: models[selected_model_name]}

    # Apply manual params
    if manual_params and selected_model_name:
        model = models[selected_model_name]
        safe_params = {}
        for k, v in manual_params.items():
            if hasattr(model, k):
                safe_params[k] = v
        if safe_params:
            model.set_params(**safe_params)

    results: List[ClusteringResult] = []

    for name, model in models.items():
        try:
            # GaussianMixture uses .fit().predict() instead of .fit_predict()
            if hasattr(model, "fit_predict"):
                labels = model.fit_predict(X)
            else:
                model.fit(X)
                labels = model.predict(X)

            metrics = _compute_clustering_metrics(X, labels)
            n_found = len(set(labels) - {-1})

            results.append(ClusteringResult(
                name=name,
                labels=labels,
                n_clusters=n_found,
                metrics=metrics,
                params=model.get_params(),
            ))
        except Exception as e:
            # Skip models that fail (e.g. MeanShift may fail on small data)
            results.append(ClusteringResult(
                name=name,
                labels=np.array([]),
                n_clusters=0,
                metrics={"error": str(e)},
            ))

    # Find best by silhouette score
    valid_results = [r for r in results if "silhouette_score" in r.metrics]
    best = max(valid_results, key=lambda r: r.metrics["silhouette_score"]) if valid_results else (results[0] if results else None)

    # Build comparison table
    comparison_rows = []
    for r in results:
        row = {"Model": r.name, "Clusters Found": r.n_clusters}
        row.update({k: round(v, 4) if isinstance(v, float) else v for k, v in r.metrics.items()})
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)

    return {
        "results": results,
        "best": best,
        "comparison": comparison_df,
        "feature_names": feature_names,
    }
