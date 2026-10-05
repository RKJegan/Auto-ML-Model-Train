"""Base search strategy interface (scaffold)."""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseSearch(ABC):
    """Abstract hyperparameter search strategy."""

    @abstractmethod
    def run(self, estimator, param_distributions, X, y, **kwargs):
        ...
