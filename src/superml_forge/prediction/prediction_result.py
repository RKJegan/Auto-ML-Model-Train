"""Result container for prediction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np


@dataclass
class PredictionResult:
    """Holds the output of prediction."""
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[Any] = None
    feature_columns: Optional[List[str]] = None
