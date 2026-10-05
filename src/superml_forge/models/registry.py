"""Model registry – central lookup for all supported models."""
from __future__ import annotations

from typing import Dict

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_classification_models(random_state: int = 42) -> Dict[str, BaseEstimator]:
    """Return a dictionary of classification model instances."""
    return {
        "Logistic Regression": LogisticRegression(random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "SVM": SVC(probability=True, random_state=random_state),
    }


def get_regression_models(random_state: int = 42) -> Dict[str, BaseEstimator]:
    """Return a dictionary of regression model instances."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(random_state=random_state),
        "Lasso Regression": Lasso(random_state=random_state),
        "Random Forest Regressor": RandomForestRegressor(random_state=random_state),
    }
