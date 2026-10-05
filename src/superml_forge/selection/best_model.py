"""Best model selection logic."""
from __future__ import annotations


def select_best(results: list, problem_type: str):
    """Select the best model from a list of results."""
    if problem_type == "classification":
        return max(results, key=lambda r: r.best_score)
    return min(results, key=lambda r: r.best_score)
