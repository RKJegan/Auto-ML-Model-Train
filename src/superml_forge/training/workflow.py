"""Training workflow – orchestrate model training."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .trainer import ModelResult, train_and_tune_models


def run_training(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    problem_type: str,
    selected_model_name: Optional[str] = None,
    manual_params: Optional[Dict[str, Any]] = None,
) -> Tuple[ModelResult, List[ModelResult], pd.DataFrame]:
    """Run the training workflow."""
    return train_and_tune_models(
        X_train=X_train,
        y_train=y_train,
        problem_type=problem_type,
        selected_model_name=selected_model_name,
        manual_params=manual_params,
    )
