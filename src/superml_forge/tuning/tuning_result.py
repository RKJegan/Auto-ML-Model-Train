"""Result container for the tuning stage."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class TuningResult:
    """Holds the output of hyperparameter tuning."""
    best_params: Optional[Dict[str, Any]] = None
    best_score: float = 0.0
    search_method: str = "randomized"
    n_iterations: int = 0
