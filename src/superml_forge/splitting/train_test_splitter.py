"""Train/test split with optional stratification."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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
