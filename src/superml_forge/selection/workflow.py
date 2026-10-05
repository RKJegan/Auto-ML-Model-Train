"""Selection workflow."""
from __future__ import annotations

from .best_model import select_best
from .ranking import rank_models
from .selection_result import SelectionResult


def run_selection(results: list, problem_type: str) -> SelectionResult:
    """Run model selection and return a SelectionResult."""
    ranked = rank_models(results, problem_type)
    best = select_best(results, problem_type)
    return SelectionResult(
        best_model_name=best.name,
        best_score=best.best_score,
        best_params=best.best_params,
        ranking=[r.name for r in ranked],
    )
