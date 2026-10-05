"""Validation workflow – orchestrate data quality checks."""
from __future__ import annotations

import pandas as pd

from .data_quality import sanitize_dataframe
from .missing_value_analyzer import get_missing_value_summary
from .validation_result import ValidationResult


def run_validation(df: pd.DataFrame) -> ValidationResult:
    """Run all validation checks and return a ValidationResult."""
    result = ValidationResult()

    missing = get_missing_value_summary(df)
    if not missing.empty:
        result.warnings.append(f"{len(missing)} column(s) have missing values.")
        result.missing_summary = missing["missing_pct"].to_dict()

    dup_count = df.duplicated().sum()
    result.duplicate_count = int(dup_count)
    if dup_count > 0:
        result.warnings.append(f"{dup_count} duplicate row(s) found.")

    return result
