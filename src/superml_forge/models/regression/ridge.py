"""Ridge Regression wrapper."""
from __future__ import annotations

from sklearn.linear_model import Ridge as _Ridge

from ..base.base_model import BaseModelWrapper


class RidgeModel(BaseModelWrapper):
    name = "Ridge Regression"

    def get_estimator(self, random_state=42, **kwargs):
        return _Ridge(random_state=random_state, **kwargs)

    def get_default_params(self):
        return {"alpha": 1.0, "fit_intercept": True}
