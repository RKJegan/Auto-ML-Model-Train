"""Cross-validation utilities (scaffold)."""
from __future__ import annotations

from sklearn.model_selection import KFold, StratifiedKFold


def get_cv_splitter(n_splits: int = 5, stratified: bool = True, random_state: int = 42):
    """Return an appropriate cross-validation splitter."""
    if stratified:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
