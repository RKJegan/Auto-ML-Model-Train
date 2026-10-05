"""Task detection workflow."""
from __future__ import annotations

import pandas as pd

from .task_classifier import detect_problem_type
from .detection_result import DetectionResult


def run_task_detection(target: pd.Series, target_column: str = "") -> DetectionResult:
    """Run task detection and return a DetectionResult."""
    problem_type = detect_problem_type(target)
    n_classes = int(target.nunique(dropna=True)) if problem_type == "classification" else 0
    return DetectionResult(
        problem_type=problem_type,
        target_column=target_column,
        n_classes=n_classes,
    )
