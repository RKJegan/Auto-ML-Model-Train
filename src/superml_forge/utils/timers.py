"""Timing utilities."""
from __future__ import annotations

import time
from contextlib import contextmanager

from .logger import get_logger

logger = get_logger()


@contextmanager
def timer(label: str = "Operation"):
    """Context manager that times a block of code."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.info(f"{label} completed in {elapsed:.2f}s")
