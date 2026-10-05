"""Model ranking utilities (scaffold)."""
from __future__ import annotations


def rank_models(results: list, problem_type: str) -> list:
    """Rank model results by their score."""
    if problem_type == "classification":
        return sorted(results, key=lambda r: r.best_score, reverse=True)
    return sorted(results, key=lambda r: r.best_score)
