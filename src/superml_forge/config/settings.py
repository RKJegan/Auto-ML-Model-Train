"""Application settings."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Settings:
    """Global application settings."""
    app_name: str = "AutoML"
    debug: bool = True
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 3
    n_iter: int = 10
    data_dir: str = "./data"
    artifacts_dir: str = "./artifacts"
    logs_dir: str = "./logs"
