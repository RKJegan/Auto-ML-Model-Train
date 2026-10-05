from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class DatasetInfo:
    """Container with basic dataset information for display in the UI."""

    shape: Tuple[int, int]
    dtypes: pd.Series
    head: pd.DataFrame


def _clean_categorical_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize string values and convert blanks to NaN for categorical columns.
    """
    cleaned = data.copy()
    for col in cleaned.columns:
        series = cleaned[col]
        series = series.astype("string").str.strip()
        series = series.replace(r"^\s*$", np.nan, regex=True)
        cleaned[col] = series
    return cleaned


def get_dataset_info(df: pd.DataFrame, head_rows: int = 5) -> DatasetInfo:
    """Return a lightweight summary of the dataset for quick inspection."""
    return DatasetInfo(
        shape=df.shape,
        dtypes=df.dtypes,
        head=df.head(head_rows),
    )


def detect_problem_type(target: pd.Series) -> str:
    """
    Infer whether the problem is classification or regression.

    Heuristic:
    - If the target dtype is object/category/bool, treat as classification.
    - Otherwise, if the number of unique values is relatively small
      compared to the dataset size, treat as classification.
    - Else, treat as regression.
    """
    if target.dtype == "O" or pd.api.types.is_categorical_dtype(target) or pd.api.types.is_bool_dtype(
        target
    ):
        return "classification"

    # Numeric target: decide based on number of unique values
    n_unique = target.nunique(dropna=True)
    n_total = len(target)
    unique_ratio = n_unique / max(n_total, 1)

    if n_unique <= 20 or unique_ratio < 0.05:
        return "classification"
    return "regression"


def build_preprocessor(
    X: pd.DataFrame,
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Create a ColumnTransformer that:
    - Imputes numeric features with median and scales them.
    - Imputes categorical features with most-frequent and one-hot encodes them.

    Returns the preprocessor and the lists of numeric and categorical feature names.
    """
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("cleaner", FunctionTransformer(_clean_categorical_values, validate=False)),
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


def train_test_split_data(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split the dataset into train and test sets.

    Uses stratification for classification problems when possible.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in the dataset.")

    cleaned_df = df.copy()
    cleaned_df[target_column] = cleaned_df[target_column].replace(r"^\s*$", np.nan, regex=True)
    if cleaned_df[target_column].isna().all():
        raise ValueError("Target column contains only missing values after cleaning.")

    # Keep all records by imputing target nulls (no row reduction).
    if problem_type == "classification":
        mode_values = cleaned_df[target_column].mode(dropna=True)
        fill_value = mode_values.iloc[0] if not mode_values.empty else "missing_target"
        cleaned_df[target_column] = cleaned_df[target_column].fillna(fill_value)
    else:
        cleaned_df[target_column] = pd.to_numeric(cleaned_df[target_column], errors="coerce")
        median_val = cleaned_df[target_column].median(skipna=True)
        if pd.isna(median_val):
            raise ValueError("Target column cannot be converted to numeric values for regression.")
        cleaned_df[target_column] = cleaned_df[target_column].fillna(median_val)

    X = cleaned_df.drop(columns=[target_column])
    y = cleaned_df[target_column]

    stratify = None
    if problem_type == "classification":
        # Only stratify if there is more than one class.
        if y.nunique(dropna=True) > 1:
            stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    return X_train, X_test, y_train, y_test


def build_full_pipeline(
    X: pd.DataFrame,
    model,
) -> Pipeline:
    """
    Attach the preprocessing ColumnTransformer in front of a model
    to create a full sklearn Pipeline.
    """
    preprocessor, _, _ = build_preprocessor(X)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline

