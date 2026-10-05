"""Result container for the task detection stage."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DetectionResult:
    """Holds the output of task detection."""
    problem_type: str  # "classification" or "regression"
    target_column: str = ""
    n_classes: int = 0
    confidence: float = 1.0
