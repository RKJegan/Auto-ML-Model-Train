"""Hyperparameter tuning – search spaces and strategies."""
from .search_space import get_classification_param_grids, get_regression_param_grids

__all__ = ["get_classification_param_grids", "get_regression_param_grids"]
