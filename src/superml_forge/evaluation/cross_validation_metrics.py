"""Cross-validation metric utilities (scaffold)."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import cross_val_score


def compute_cv_scores(estimator, X, y, scoring="accuracy", cv=5, n_jobs=-1) -> dict:
    """Compute cross-validation scores and return summary stats."""
    scores = cross_val_score(estimator, X, y, scoring=scoring, cv=cv, n_jobs=n_jobs)
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "scores": scores.tolist(),
    }
