"""Unsupervised workflow orchestrator."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


import pandas as pd


def run_unsupervised_workflow(
    df: pd.DataFrame,
    task: str,
    columns: Optional[List[str]] = None,
    model_name: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the unsupervised learning workflow.

    Args:
        df: Input DataFrame
        task: One of "clustering", "dimensionality_reduction", "anomaly_detection"
        columns: Columns to use (None = all)
        model_name: Specific algorithm name (None = run all)
        params: Manual hyperparameters

    Returns:
        Dict with results depending on the task.
    """
    from .preprocessor import preprocess_for_unsupervised

    X_scaled, feature_names, df_clean = preprocess_for_unsupervised(df, columns)

    if task == "clustering":
        from .clustering.trainer import train_clustering_models
        return train_clustering_models(X_scaled, feature_names, model_name, params)
    elif task == "dimensionality_reduction":
        from .dimensionality_reduction.reducer import run_reduction
        return run_reduction(X_scaled, feature_names, model_name, params)
    elif task == "anomaly_detection":
        from .anomaly_detection.detector import run_anomaly_detection
        return run_anomaly_detection(X_scaled, feature_names, model_name, params)
    else:
        raise ValueError(f"Unknown unsupervised task: {task}")
