"""Registry of anomaly detection algorithms."""
from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


def get_anomaly_models(
    contamination: float = 0.1, random_state: int = 42
) -> Dict[str, Any]:
    """Return a dictionary of anomaly detection model instances."""
    return {
        "Isolation Forest": IsolationForest(
            contamination=contamination, random_state=random_state, n_estimators=100
        ),
        "Local Outlier Factor": LocalOutlierFactor(
            n_neighbors=20, contamination=contamination, novelty=False
        ),
        "One-Class SVM": OneClassSVM(kernel="rbf", nu=contamination, gamma="scale"),
    }


def get_anomaly_param_options() -> Dict[str, Dict[str, Any]]:
    """Return hyperparameter options for each anomaly detection algorithm."""
    return {
        "Isolation Forest": {
            "n_estimators": {"type": "int", "min": 50, "max": 500, "default": 100},
            "contamination": {"type": "float", "min": 0.01, "max": 0.5, "default": 0.1},
        },
        "Local Outlier Factor": {
            "n_neighbors": {"type": "int", "min": 5, "max": 100, "default": 20},
            "contamination": {"type": "float", "min": 0.01, "max": 0.5, "default": 0.1},
        },
        "One-Class SVM": {
            "kernel": {"type": "select", "options": ["rbf", "linear", "poly", "sigmoid"], "default": "rbf"},
            "nu": {"type": "float", "min": 0.01, "max": 0.5, "default": 0.1},
            "gamma": {"type": "select", "options": ["scale", "auto"], "default": "scale"},
        },
    }
