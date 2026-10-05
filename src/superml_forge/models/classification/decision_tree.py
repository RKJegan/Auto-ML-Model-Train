"""Decision Tree Classifier wrapper."""
from __future__ import annotations

from sklearn.tree import DecisionTreeClassifier as _DT

from ..base.base_model import BaseModelWrapper


class DecisionTreeModel(BaseModelWrapper):
    name = "Decision Tree"

    def get_estimator(self, random_state=42, **kwargs):
        return _DT(random_state=random_state, **kwargs)

    def get_default_params(self):
        return {"max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1}
