"""Metrics domain object."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class MetricsReport:
    """Holds evaluation metrics."""
    metrics: Dict[str, float] = field(default_factory=dict)
    problem_type: str = ""
