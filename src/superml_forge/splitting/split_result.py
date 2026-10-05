"""Result container for the splitting stage."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class SplitResult:
    """Holds the output of data splitting."""
    X_train: pd.DataFrame | None = None
    X_test: pd.DataFrame | None = None
    y_train: pd.Series | None = None
    y_test: pd.Series | None = None
    test_size: float = 0.2
    is_stratified: bool = False
