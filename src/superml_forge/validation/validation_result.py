"""Result container for the validation stage."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ValidationResult:
    """Holds the output of data validation."""
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    missing_summary: Dict[str, float] = field(default_factory=dict)
    duplicate_count: int = 0
