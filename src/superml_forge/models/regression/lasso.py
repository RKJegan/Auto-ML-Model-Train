"""Lasso Regression wrapper."""
from __future__ import annotations

from sklearn.linear_model import Lasso as _Lasso

from ..base.base_model import BaseModelWrapper


class LassoModel(BaseModelWrapper):
    name = "Lasso Regression"

    def get_estimator(self, random_state=42, **kwargs):
        return _Lasso(random_state=random_state, **kwargs)

    def get_default_params(self):
        return {"alpha": 1.0, "max_iter": 1000}
