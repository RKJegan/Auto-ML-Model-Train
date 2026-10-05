"""Ingestion workflow – orchestrate file loading."""
from __future__ import annotations

from typing import Optional



from .loader import load_data
from .format_detector import detect_extension
from .ingestion_result import IngestionResult


def run_ingestion(uploaded_file, sheet_name: Optional[str] = None) -> IngestionResult:
    """Run the full ingestion workflow and return an IngestionResult."""
    file_type = detect_extension(uploaded_file.name)
    df = load_data(uploaded_file, sheet_name=sheet_name)
    return IngestionResult(
        dataframe=df,
        file_name=uploaded_file.name,
        file_type=file_type,
        sheet_name=sheet_name,
    )
