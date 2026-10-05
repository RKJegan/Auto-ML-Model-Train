"""Numerical scaling utilities."""
from __future__ import annotations

from sklearn.preprocessing import StandardScaler


def get_standard_scaler() -> StandardScaler:
    """Return a StandardScaler instance."""
    return StandardScaler()
