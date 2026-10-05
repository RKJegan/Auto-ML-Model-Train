"""Random Forest Classifier wrapper."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier as _RF

from ..base.base_model import BaseModelWrapper


class RandomForestClassifierModel(BaseModelWrapper):
    name = "Random Forest"

    def get_estimator(self, random_state=42, **kwargs):
        return _RF(random_state=random_state, **kwargs)

    def get_default_params(self):
        return {"n_estimators": 100, "max_depth": None, "min_samples_split": 2}
