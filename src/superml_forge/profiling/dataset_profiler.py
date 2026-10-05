"""Dataset-level profiling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd


@dataclass
class DatasetInfo:
    """Container with basic dataset information for display in the UI."""
    shape: Tuple[int, int]
    dtypes: pd.Series
    head: pd.DataFrame


def get_dataset_info(df: pd.DataFrame, head_rows: int = 5) -> DatasetInfo:
    """Return a lightweight summary of the dataset for quick inspection."""
    return DatasetInfo(
        shape=df.shape,
        dtypes=df.dtypes,
        head=df.head(head_rows),
    )
