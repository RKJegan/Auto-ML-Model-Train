"""Random Forest Regressor wrapper."""
from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor as _RFR

from ..base.base_model import BaseModelWrapper


class RandomForestRegressorModel(BaseModelWrapper):
    name = "Random Forest Regressor"

    def get_estimator(self, random_state=42, **kwargs):
        return _RFR(random_state=random_state, **kwargs)

    def get_default_params(self):
        return {"n_estimators": 100, "max_depth": None, "min_samples_split": 2}
