"""Missing value imputation strategies."""
from __future__ import annotations

from sklearn.impute import SimpleImputer


def get_numeric_imputer(strategy: str = "median") -> SimpleImputer:
    """Return an imputer for numeric features."""
    return SimpleImputer(strategy=strategy)


def get_categorical_imputer(strategy: str = "most_frequent", fill_value: str = "missing") -> SimpleImputer:
    """Return an imputer for categorical features."""
    return SimpleImputer(strategy=strategy, fill_value=fill_value)
