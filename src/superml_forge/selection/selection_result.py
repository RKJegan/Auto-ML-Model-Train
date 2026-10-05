"""Result container for the selection stage."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SelectionResult:
    """Holds the output of model selection."""
    best_model_name: str = ""
    best_score: float = 0.0
    best_params: Optional[Dict[str, Any]] = None
    ranking: Optional[List[Any]] = None
