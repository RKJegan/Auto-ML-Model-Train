"""Base model wrapper interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from sklearn.base import BaseEstimator


class BaseModelWrapper(ABC):
    """Abstract wrapper around an sklearn estimator."""

    name: str  # Human-readable model name (set by subclasses as a class attribute)

    @abstractmethod
    def get_estimator(self, random_state: int = 42, **kwargs) -> BaseEstimator:
        """Return a fresh estimator instance."""
        ...

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Return sensible default hyperparameters."""
        ...
