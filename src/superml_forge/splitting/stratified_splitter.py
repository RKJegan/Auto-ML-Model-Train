"""Stratified splitting utilities (scaffold)."""
from __future__ import annotations

from sklearn.model_selection import StratifiedShuffleSplit


def get_stratified_split(n_splits: int = 1, test_size: float = 0.2, random_state: int = 42):
    """Return a StratifiedShuffleSplit object."""
    return StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
