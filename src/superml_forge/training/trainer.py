"""Core training and tuning logic."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

from ..preprocessing.pipeline_builder import build_full_pipeline
from .model_runner import choose_models


@dataclass
class ModelResult:
    """Result of training a single model."""
    name: str
    best_estimator: BaseEstimator
    best_score: float
    best_params: Dict[str, Any]


def train_and_tune_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    problem_type: str,
    selected_model_name: Optional[str] = None,
    manual_params: Optional[Dict[str, Any]] = None,
    n_iter: int = 10,
    cv_folds: int = 3,
    random_state: int = 42,
) -> Tuple[ModelResult, List[ModelResult], pd.DataFrame]:
    """
    Train and tune models using RandomizedSearchCV.

    Returns:
    - best_model_result: the best-performing model and its information.
    - all_model_results: list of results for each tried model.
    - comparison_df: table summarizing performance of all models.
    """
    models, param_grids = choose_models(
        problem_type, selected_model_name=selected_model_name, random_state=random_state
    )

    scoring = "accuracy" if problem_type == "classification" else "neg_root_mean_squared_error"

    all_results: List[ModelResult] = []

    for name, base_model in models.items():
        use_manual = manual_params is not None and selected_model_name == name
        if use_manual:
            base_model = base_model.set_params(**manual_params)

        pipeline = build_full_pipeline(X_train, base_model)
        param_distributions = {} if use_manual else param_grids.get(name, {})

        if param_distributions:
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_distributions,
                n_iter=n_iter,
                scoring=scoring,
                n_jobs=-1,
                cv=cv_folds,
                random_state=random_state,
                refit=True,
            )
            search.fit(X_train, y_train)
            best_estimator = search.best_estimator_
            if problem_type == "classification":
                best_score = float(search.best_score_)
            else:
                best_score = float(-search.best_score_)
            best_params = dict(search.best_params_)
        else:
            pipeline.fit(X_train, y_train)
            best_estimator = pipeline
            cv_scores = cross_val_score(
                best_estimator,
                X_train,
                y_train,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1,
            )
            if problem_type == "classification":
                best_score = float(np.mean(cv_scores))
            else:
                best_score = float(-np.mean(cv_scores))
            best_params = manual_params if use_manual else {}

        all_results.append(
            ModelResult(
                name=name,
                best_estimator=best_estimator,
                best_score=best_score,
                best_params=best_params,
            )
        )

    if problem_type == "classification":
        best_model = max(all_results, key=lambda r: r.best_score)
    else:
        best_model = min(all_results, key=lambda r: r.best_score)

    comparison_rows = []
    if problem_type == "classification":
        metric_label = "CV Accuracy"
        for r in all_results:
            comparison_rows.append(
                {"Model": r.name, metric_label: r.best_score, "Best Params": r.best_params}
            )
    else:
        metric_label = "CV RMSE"
        for r in all_results:
            comparison_rows.append(
                {"Model": r.name, metric_label: r.best_score, "Best Params": r.best_params}
            )

    comparison_df = pd.DataFrame(comparison_rows)

    return best_model, all_results, comparison_df
