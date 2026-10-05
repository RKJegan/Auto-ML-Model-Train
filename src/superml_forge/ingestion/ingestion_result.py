"""Result container for the ingestion stage."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd


@dataclass
class IngestionResult:
    """Holds the output of the data ingestion step."""
    dataframe: pd.DataFrame
    file_name: str = ""
    file_type: str = ""
    sheet_name: Optional[str] = None
    row_count: int = 0
    column_count: int = 0
    column_names: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.dataframe is not None:
            self.row_count = len(self.dataframe)
            self.column_count = len(self.dataframe.columns)
            self.column_names = self.dataframe.columns.tolist()
