"""Randomized search wrapper."""
from __future__ import annotations

from sklearn.model_selection import RandomizedSearchCV


def create_randomized_search(
    estimator, param_distributions, n_iter=10, scoring="accuracy", cv=3, random_state=42, n_jobs=-1
):
    """Create and return a RandomizedSearchCV instance."""
    return RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        refit=True,
    )
