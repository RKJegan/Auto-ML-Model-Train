"""Dimensionality reduction logic."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .registry import get_reduction_models


@dataclass
class ReductionResult:
    """Result of a single reduction model."""
    name: str
    transformed: np.ndarray
    n_components: int
    metrics: Dict[str, float] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)


def run_reduction(
    X: np.ndarray,
    feature_names: List[str],
    selected_model_name: Optional[str] = None,
    manual_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run dimensionality reduction and compute metrics.

    Returns dict with results for each model.
    """
    n_components = 2
    if manual_params and "n_components" in manual_params:
        n_components = manual_params["n_components"]

    # Ensure n_components doesn't exceed n_features
    n_components = min(n_components, X.shape[1] - 1) if X.shape[1] > 2 else 2
    n_components = max(n_components, 2)

    models = get_reduction_models(n_components=n_components)

    if selected_model_name:
        if selected_model_name not in models:
            raise ValueError(f"Unknown reduction model: {selected_model_name}")
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

    results: List[ReductionResult] = []

    for name, model in models.items():
        try:
            transformed = model.fit_transform(X)

            metrics = {"n_components": n_components}

            # PCA and TruncatedSVD have explained_variance_ratio_
            if hasattr(model, "explained_variance_ratio_"):
                evr = model.explained_variance_ratio_
                metrics["total_explained_variance"] = float(np.sum(evr))
                for i, v in enumerate(evr):
                    metrics[f"PC{i+1}_variance"] = float(v)

            results.append(ReductionResult(
                name=name,
                transformed=transformed,
                n_components=n_components,
                metrics=metrics,
                params=model.get_params(),
            ))
        except Exception as e:
            results.append(ReductionResult(
                name=name,
                transformed=np.array([]),
                n_components=0,
                metrics={"error": str(e)},
            ))

    # Best = highest explained variance (only for PCA/SVD); otherwise first
    valid = [r for r in results if "total_explained_variance" in r.metrics]
    best = max(valid, key=lambda r: r.metrics["total_explained_variance"]) if valid else (results[0] if results else None)

    comparison_rows = []
    for r in results:
        row = {"Model": r.name}
        row.update({k: round(v, 4) if isinstance(v, float) else v for k, v in r.metrics.items()})
        comparison_rows.append(row)

    return {
        "results": results,
        "best": best,
        "comparison": pd.DataFrame(comparison_rows),
        "feature_names": feature_names,
    }
