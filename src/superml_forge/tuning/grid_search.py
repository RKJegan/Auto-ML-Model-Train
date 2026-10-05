"""Grid search wrapper (scaffold)."""
from __future__ import annotations

from sklearn.model_selection import GridSearchCV


def create_grid_search(estimator, param_grid, scoring="accuracy", cv=3, n_jobs=-1):
    """Create and return a GridSearchCV instance."""
    return GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
    )
