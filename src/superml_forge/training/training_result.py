"""Result container for training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from sklearn.base import BaseEstimator


@dataclass
class ModelResult:
    """Result of training a single model."""
    name: str
    best_estimator: BaseEstimator
    best_score: float
    best_params: Dict[str, Any]
