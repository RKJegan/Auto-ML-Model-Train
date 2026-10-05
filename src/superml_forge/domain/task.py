"""Task domain object."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Task:
    """Describes the ML task."""
    problem_type: str  # "classification" or "regression"
    target_column: str
    n_classes: int = 0
