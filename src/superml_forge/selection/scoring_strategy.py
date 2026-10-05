"""Scoring strategies for model selection (scaffold)."""
from __future__ import annotations


def get_scoring_metric(problem_type: str) -> str:
    """Return the primary scoring metric for the given problem type."""
    if problem_type == "classification":
        return "accuracy"
    return "neg_root_mean_squared_error"
