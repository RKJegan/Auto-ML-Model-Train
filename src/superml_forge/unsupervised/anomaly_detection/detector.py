"""Anomaly detection logic."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .registry import get_anomaly_models


@dataclass
class AnomalyResult:
    """Result of a single anomaly detection model."""
    name: str
    labels: np.ndarray        # 1 = normal, -1 = anomaly
    scores: np.ndarray        # anomaly scores (lower = more anomalous)
    n_anomalies: int
    anomaly_pct: float
    metrics: Dict[str, float] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)


def run_anomaly_detection(
    X: np.ndarray,
    feature_names: List[str],
    selected_model_name: Optional[str] = None,
    manual_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run anomaly detection models.

    Returns dict with results for each model.
    """
    contamination = 0.1
    if manual_params and "contamination" in manual_params:
        contamination = manual_params["contamination"]
    if manual_params and "nu" in manual_params:
        contamination = manual_params["nu"]

    models = get_anomaly_models(contamination=contamination)

    if selected_model_name:
        if selected_model_name not in models:
            raise ValueError(f"Unknown anomaly model: {selected_model_name}")
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

    results: List[AnomalyResult] = []

    for name, model in models.items():
        try:
            if name == "Local Outlier Factor":
                # LOF with novelty=False uses fit_predict
                labels = model.fit_predict(X)
                scores = model.negative_outlier_factor_
            else:
                model.fit(X)
                labels = model.predict(X)
                if hasattr(model, "decision_function"):
                    scores = model.decision_function(X)
                elif hasattr(model, "score_samples"):
                    scores = model.score_samples(X)
                else:
                    scores = np.zeros(len(X))

            n_anomalies = int(np.sum(labels == -1))
            anomaly_pct = round(n_anomalies / len(X) * 100, 2)

            metrics = {
                "n_anomalies": n_anomalies,
                "anomaly_pct": anomaly_pct,
                "n_normal": int(np.sum(labels == 1)),
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
            }

            results.append(AnomalyResult(
                name=name,
                labels=labels,
                scores=scores,
                n_anomalies=n_anomalies,
                anomaly_pct=anomaly_pct,
                metrics=metrics,
                params=model.get_params(),
            ))
        except Exception as e:
            results.append(AnomalyResult(
                name=name,
                labels=np.array([]),
                scores=np.array([]),
                n_anomalies=0,
                anomaly_pct=0.0,
                metrics={"error": str(e)},
            ))

    # Best = model that found anomalies closest to expected contamination
    valid = [r for r in results if r.labels.size > 0]
    best = valid[0] if valid else (results[0] if results else None)

    comparison_rows = []
    for r in results:
        comparison_rows.append({
            "Model": r.name,
            "Anomalies Found": r.n_anomalies,
            "Anomaly %": r.anomaly_pct,
            "Mean Score": round(r.metrics.get("mean_score", 0), 4),
        })

    return {
        "results": results,
        "best": best,
        "comparison": pd.DataFrame(comparison_rows),
        "feature_names": feature_names,
    }
