"""Logistic Regression wrapper."""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression as _LR

from ..base.base_model import BaseModelWrapper


class LogisticRegressionModel(BaseModelWrapper):
    name = "Logistic Regression"

    def get_estimator(self, random_state=42, **kwargs):
        return _LR(random_state=random_state, **kwargs)

    def get_default_params(self):
        return {"C": 1.0, "max_iter": 1000, "penalty": "l2", "solver": "lbfgs"}
