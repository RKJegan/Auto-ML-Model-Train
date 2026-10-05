"""Prediction domain object."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class PredictionOutput:
    """Holds prediction results."""
    value: Any
    probability: Optional[float] = None
    model_name: str = ""
