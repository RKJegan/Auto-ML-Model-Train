"""Tuning workflow (scaffold)."""
from __future__ import annotations

from .tuning_result import TuningResult


def run_tuning(estimator, param_distributions, X, y, **kwargs) -> TuningResult:
    """Run hyperparameter tuning and return a TuningResult."""
    from .randomized_search import create_randomized_search
    search = create_randomized_search(estimator, param_distributions, **kwargs)
    search.fit(X, y)
    return TuningResult(
        best_params=dict(search.best_params_),
        best_score=float(search.best_score_),
        search_method="randomized",
        n_iterations=kwargs.get("n_iter", 10),
    )
