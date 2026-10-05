"""Result container for the preprocessing stage."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd


@dataclass
class PreprocessingResult:
    """Holds the output of preprocessing."""
    X_transformed: pd.DataFrame | None = None
    numeric_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    n_features_out: int = 0
