"""Profiling workflow – orchestrate dataset profiling."""
from __future__ import annotations

import pandas as pd

from .dataset_profiler import DatasetInfo, get_dataset_info
from .column_profiler import profile_column
from .profile_result import ProfileResult


def run_profiling(df: pd.DataFrame) -> ProfileResult:
    """Run profiling on the DataFrame."""
    result = ProfileResult(n_rows=len(df), n_columns=len(df.columns))
    for col in df.columns:
        result.column_profiles[col] = profile_column(df[col])
    return result
