"""Result container for the evaluation stage."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class EvaluationResult:
    """Holds the output of model evaluation."""
    model_name: str = ""
    problem_type: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
