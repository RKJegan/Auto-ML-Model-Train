"""Model selection helper for the training loop."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from sklearn.base import BaseEstimator

from ..models.registry import get_classification_models, get_regression_models
from ..tuning.search_space import get_classification_param_grids, get_regression_param_grids


def choose_models(
    problem_type: str, selected_model_name: Optional[str] = None, random_state: int = 42
) -> Tuple[Dict[str, BaseEstimator], Dict[str, Dict[str, Any]]]:
    """Return the model dict and corresponding param grids for the chosen problem type."""
    if problem_type == "classification":
        all_models = get_classification_models(random_state=random_state)
        grids = get_classification_param_grids()
    else:
        all_models = get_regression_models(random_state=random_state)
        grids = get_regression_param_grids()

    if selected_model_name is not None:
        chosen_models = {selected_model_name: all_models[selected_model_name]}
        chosen_grids = {selected_model_name: grids.get(selected_model_name, {})}
        return chosen_models, chosen_grids

    return all_models, grids
