"""Splitting workflow."""
from __future__ import annotations

import pandas as pd

from .train_test_splitter import train_test_split_data
from .split_result import SplitResult


def run_splitting(
    df: pd.DataFrame, target_column: str, problem_type: str, test_size: float = 0.2
) -> SplitResult:
    """Run the splitting workflow and return a SplitResult."""
    X_train, X_test, y_train, y_test = train_test_split_data(df, target_column, problem_type, test_size)
    return SplitResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        test_size=test_size,
        is_stratified=(problem_type == "classification"),
    )
