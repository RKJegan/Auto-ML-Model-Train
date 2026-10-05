"""SVM Classifier wrapper."""
from __future__ import annotations

from sklearn.svm import SVC as _SVC

from ..base.base_model import BaseModelWrapper


class SVMModel(BaseModelWrapper):
    name = "SVM"

    def get_estimator(self, random_state=42, **kwargs):
        return _SVC(probability=True, random_state=random_state, **kwargs)

    def get_default_params(self):
        return {"C": 1.0, "kernel": "rbf", "gamma": "scale"}
