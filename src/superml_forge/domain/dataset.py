"""Dataset domain object."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd


@dataclass
class Dataset:
    """Represents a loaded dataset."""
    dataframe: pd.DataFrame
    name: str = ""
    shape: Tuple[int, int] = (0, 0)
    column_names: Optional[List[str]] = None

    def __post_init__(self):
        if self.dataframe is not None:
            self.shape = self.dataframe.shape
            self.column_names = self.dataframe.columns.tolist()
