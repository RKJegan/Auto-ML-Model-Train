"""Result container for feature engineering."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class FeatureResult:
    """Holds the output of feature engineering."""
    selected_features: List[str] = field(default_factory=list)
    dropped_features: List[str] = field(default_factory=list)
    n_features_in: int = 0
    n_features_out: int = 0
