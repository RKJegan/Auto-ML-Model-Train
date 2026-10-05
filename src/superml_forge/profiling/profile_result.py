"""Result container for the profiling stage."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ProfileResult:
    """Holds the output of dataset profiling."""
    n_rows: int = 0
    n_columns: int = 0
    column_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
