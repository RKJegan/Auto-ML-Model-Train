"""Linear Regression wrapper."""
from __future__ import annotations

from sklearn.linear_model import LinearRegression as _LR

from ..base.base_model import BaseModelWrapper


class LinearRegressionModel(BaseModelWrapper):
    name = "Linear Regression"

    def get_estimator(self, random_state=42, **kwargs):
        return _LR(**kwargs)

    def get_default_params(self):
        return {"fit_intercept": True}
