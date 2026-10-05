"""General helper functions."""
from __future__ import annotations

from typing import Any, Dict


def safe_dict_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dicts, with override taking precedence."""
    result = base.copy()
    result.update(override)
    return result
