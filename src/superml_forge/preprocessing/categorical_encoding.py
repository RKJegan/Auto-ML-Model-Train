"""Categorical encoding utilities."""
from __future__ import annotations

from sklearn.preprocessing import OneHotEncoder


def get_onehot_encoder(handle_unknown: str = "ignore") -> OneHotEncoder:
    """Return a one-hot encoder with sensible defaults."""
    return OneHotEncoder(handle_unknown=handle_unknown)
