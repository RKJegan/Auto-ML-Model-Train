"""Preprocessing – imputation, encoding, scaling, and pipeline construction."""
from .pipeline_builder import build_preprocessor, build_full_pipeline
from .preprocessing_result import PreprocessingResult

__all__ = ["build_preprocessor", "build_full_pipeline", "PreprocessingResult"]
