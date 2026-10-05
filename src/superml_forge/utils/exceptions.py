"""Custom exception classes."""
from __future__ import annotations


class SuperMLError(Exception):
    """Base exception for SuperML-Forge."""


class DataIngestionError(SuperMLError):
    """Raised when data ingestion fails."""


class ValidationError(SuperMLError):
    """Raised when data validation fails."""


class PreprocessingError(SuperMLError):
    """Raised when preprocessing fails."""


class TrainingError(SuperMLError):
    """Raised when model training fails."""


class PredictionError(SuperMLError):
    """Raised when prediction fails."""
