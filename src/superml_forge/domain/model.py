"""Model domain object."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class TrainedModel:
    """Represents a trained model and its metadata."""
    name: str
    estimator: Any
    params: Optional[Dict[str, Any]] = None
    score: float = 0.0
