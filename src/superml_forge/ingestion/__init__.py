"""Data ingestion – load datasets from multiple file formats."""
from .format_detector import SUPPORTED_EXTENSIONS, detect_extension, list_excel_sheets
from .file_handler import standardize_column_names
from .loader import load_data
from .ingestion_result import IngestionResult

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "detect_extension",
    "list_excel_sheets",
    "standardize_column_names",
    "load_data",
    "IngestionResult",
]
